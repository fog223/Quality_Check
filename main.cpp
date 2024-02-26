// local
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
// liblas
#include <liblas/liblas.hpp>
// pcl
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/pca.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/random_sample.h>

/// 读取las文件，为了避免大数值坐标的影响，进行偏移
/// \param las_path las文件路径
/// \param cloud 获得的点云
/// \return
bool read_las(std::string las_path, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	double& x_offset, double& y_offset, double& z_offset)
{
	// 打开 LAS 文件
	std::ifstream ifs(las_path, std::ios::in | std::ios::binary);
	if (!ifs.is_open())
	{
		std::cerr << "Error: Failed to open file." << std::endl;
		return 1;
	}

	// 创建 libLAS 读取器
	liblas::ReaderFactory reader_factory;
	liblas::Reader reader = reader_factory.CreateWithStream(ifs);

	// 获取 LAS 文件头
	const liblas::Header& header = reader.GetHeader();

	if (x_offset == 0 && y_offset == 0 && z_offset == 0)
	{
		x_offset = header.GetOffsetX();
		y_offset = header.GetOffsetY();
		z_offset = header.GetOffsetZ();
	}

	// 读取点云数据
	while (reader.ReadNextPoint())
	{
		liblas::Point const& point = reader.GetPoint();
		// 处理点云数据
		cloud->push_back(pcl::PointXYZ(point.GetX() - x_offset, point.GetY() - y_offset, point.GetZ() - z_offset));
	}

	// 关闭文件
	ifs.close();

	return true;
}

/// 单个格网
struct Single_Grim
{
	std::vector<int> idx_flight1;
	std::vector<int> idx_flight2;
};

// Hash
using Array2I = std::array<int, 2>;
struct HashArray2I
{
	std::size_t operator()(const Array2I& rhs) const noexcept
	{
		return (std::hash<int>()(rhs[0])) ^ (std::hash<int>()(rhs[1]));
	}
};

/// 输入为：相邻两航带点云数据
/// 1、计算两条航带各自覆盖面积及点密度
/// 2、计算两条航带最小及最小重叠
/// 3、对两条航带的重叠部分进行高程差异分析
/// \param flight1 航线1点云数据
/// \param flight2 航线2点云数据
/// \return
bool Quality_Check(pcl::PointCloud<pcl::PointXYZ>::Ptr flight1, pcl::PointCloud<pcl::PointXYZ>::Ptr flight2)
{
	// 将两航带数据融合，计算航线方向，并以航线方向为X轴，垂直于航线方向为Y轴构建格网
	// 格网大小自适应计算，重叠度计算方式为：对重叠区域中的部分，相同X值的为一列，列长为重叠大小
	// 高差分析方式为：在重叠区中，航线1采样N个点，找出这N个点在航线2中对应高程值，
	// 当前采用最近邻半径检索法，并将半径内的所有点的平均高程为插值高程

	// 哈希表方式存储网格，加快数据检索速率
	std::unordered_map<Array2I, Single_Grim, HashArray2I> grim;

	/// 航线方向计算
	// 创建体素降采样对象
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < flight1->size(); i++)
	{
		cloud_temp->push_back(pcl::PointXYZ(flight1->points[i].x, flight1->points[i].y, 0.0f));
	}
	for (int i = 0; i < flight2->size(); i++)
	{
		cloud_temp->push_back(pcl::PointXYZ(flight2->points[i].x, flight2->points[i].y, 0.0f));
	}

	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud_temp);
	pcl::PointXYZ min_pt, max_pt;
	pcl::getMinMax3D(*cloud_temp, min_pt, max_pt);
	auto LeafSize = sqrt((max_pt.x - min_pt.x) * (max_pt.y - min_pt.y)) / 100; // 自适应计算体素大小，大约取10000个点进行pca
	sor.setLeafSize(LeafSize, LeafSize, LeafSize); // 设置体素大小

	// 执行体素降采样
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
	sor.filter(*cloud_downsampled);

	// 创建 PCA 对象
	pcl::PCA<pcl::PointXYZ> pca;
	pca.setInputCloud(cloud_downsampled);

	// 提取主方向
	auto first_principal = pca.getEigenVectors().col(0); // x
	auto second_principal = pca.getEigenVectors().col(1); // y

	/// 自适应格网尺寸（每个格网大于16个点）
	auto area_pre = cloud_downsampled->size() * LeafSize * LeafSize; // 覆盖面积预估
	auto grim_size = int(sqrt(16.0 / (0.8 * (flight1->size() + flight2->size()) / area_pre))) + 1; // 格网划分后，每个格网内大概多余16个点
	std::cout << "grim_size: " << grim_size << std::endl;

	// 格网划分的最小x,最小y
	float min_projection_x = cloud_temp->points[0].getVector3fMap().dot(first_principal);
	float min_projection_y = cloud_temp->points[0].getVector3fMap().dot(second_principal);
	for (const auto& pt : cloud_temp->points)
	{
		auto projection_x = pt.getVector3fMap().dot(first_principal);
		auto projection_y = pt.getVector3fMap().dot(second_principal);
		if (pt.getVector3fMap().dot(first_principal) < min_projection_x)
			min_projection_x = projection_x;
		if (pt.getVector3fMap().dot(second_principal) < min_projection_y)
			min_projection_y = projection_y;
	}

	// 将点落入对应格网并存储
	Array2I idx;
	for (int i = 0; i < flight1->size(); i++)
	{
		idx[0] = int((flight1->points[i].getVector3fMap().dot(first_principal) - min_projection_x) / grim_size);
		idx[1] = int((flight1->points[i].getVector3fMap().dot(second_principal) - min_projection_y) / grim_size);

		// Never appeared
		Single_Grim single_grim;
		if (grim.count(idx) == 0)
		{
			single_grim.idx_flight1.push_back(i);
			grim.insert({ idx, single_grim });
		}
		else
		{
			grim[idx].idx_flight1.push_back(i);
		}
	}

	for (int i = 0; i < flight2->size(); i++)
	{
		idx[0] = int(fabs(flight2->points[i].getVector3fMap().dot(first_principal) - min_projection_x) / grim_size);
		idx[1] = int(fabs(flight2->points[i].getVector3fMap().dot(second_principal) - min_projection_y) / grim_size);

		// Never appeared
		Single_Grim single_grim;
		if (grim.count(idx) == 0)
		{
			single_grim.idx_flight2.push_back(i);
			grim.insert({ idx, single_grim });
		}
		else
		{
			grim[idx].idx_flight2.push_back(i);
		}
	}

	// 统计航线1及航线2覆盖面积
	float flight1_grim_num = 0;
	float flight2_grim_num = 0;
	// 航线重叠区按x值相同的存储，便于计算列长度
	std::unordered_map<int, std::pair<int, int>> overlap_line;
	// 航线重叠区点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1_overlap(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2_overlap(new pcl::PointCloud<pcl::PointXYZ>);
	for (auto it = grim.begin(); it != grim.end(); ++it)
	{
		// 仅有航线1点的格网
		if (it->second.idx_flight1.size() > 0 && it->second.idx_flight2.empty())
			flight1_grim_num++;
			// 仅有航线2点的格网
		else if (it->second.idx_flight1.empty() && it->second.idx_flight2.size() > 0)
			flight2_grim_num++;
			// 重叠格网
		else
		{
			flight1_grim_num++;
			flight2_grim_num++;

			if (overlap_line.count(it->first[0]) == 0)
			{
				overlap_line.insert({ it->first[0], std::pair<int, int>(it->first[1], it->first[1]) });
			}
			else
			{
				if (it->first[1] < overlap_line[it->first[0]].first)
					overlap_line[it->first[0]].first = it->first[1];
				else if (it->first[1] > overlap_line[it->first[0]].second)
					overlap_line[it->first[0]].second = it->first[1];
			}

			for (const auto& idx : it->second.idx_flight1)
			{
				cloud1_overlap->push_back(flight1->points[idx]);
			}
			for (const auto& idx : it->second.idx_flight2)
			{
				cloud2_overlap->push_back(flight2->points[idx]);
			}
		}
	}

	// 计算最小重叠与最大重叠
	float min_overlap = overlap_line.begin()->second.second - overlap_line.begin()->second.first;
	float max_overlap = overlap_line.begin()->second.second - overlap_line.begin()->second.first;
	for (auto it = overlap_line.begin(); it != overlap_line.end(); ++it)
	{
		auto length = it->second.second - it->second.first;
		if (length < min_overlap)
			min_overlap = length;
		else if (length > max_overlap)
			max_overlap = length;
	}

	/// 高程差异分析
	pcl::PointCloud<pcl::PointXYZ>::Ptr N_sample(new pcl::PointCloud<pcl::PointXYZ>);
	// 重复区域中选取N个点计算高程差异
	int sampleSize = 10000;

	// 创建随机采样对象
	pcl::RandomSample<pcl::PointXYZ> randomSample;
	randomSample.setInputCloud(cloud1_overlap);
	randomSample.setSample(sampleSize);

	// 执行随机采样
	randomSample.filter(*N_sample);

	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud2_overlap);
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;

	// 计算平均高程差
	std::vector<float> delta_H;
	float avg_H = 0.0;
	for (const auto& pt : N_sample->points)
	{
		// 检索半径为格网尺寸的一半
		if (kdtree.radiusSearch(pt, float(grim_size) / 2, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
		{
			// 高程内插：该点半径内的所有点的平均高程
			float pt_h = 0.0;
			for (const auto& knn_idx : pointIdxRadiusSearch)
			{
				pt_h += cloud2_overlap->points[knn_idx].z;
			}
			pt_h /= pointIdxRadiusSearch.size();

			auto delta_h = pt.z - pt_h;
			avg_H += delta_h;
			delta_H.push_back(delta_h);
		}
	}
	avg_H /= delta_H.size();

	// 计算高差标准差
	float H = 0;
	for (int i = 0; i < delta_H.size(); i++)
	{
		H += pow(delta_H[i] - avg_H, 2);
	}
	auto sigma = sqrt(H / delta_H.size());

	/// 参数计算及输出
	float flight1_area = flight1_grim_num * grim_size * grim_size;
	float flight1_density = flight1->size() / flight1_area;
	float flight2_area = flight2_grim_num * grim_size * grim_size;
	float flight2_density = flight2->size() / flight2_area;
	std::cout << "flight1 area: " << flight1_area << std::endl;
	std::cout << "flight1 density: " << flight1_density << std::endl;
	std::cout << "flight2 area: " << flight2_area << std::endl;
	std::cout << "flight2 density: " << flight2_density << std::endl;
	std::cout << "min_overlap: " << min_overlap << std::endl;
	std::cout << "max_overlap: " << max_overlap << std::endl;
	std::cout << "avg_H: " << avg_H << std::endl;
	std::cout << "sigma: " << sigma << std::endl;

	return true;
}

int main()
{
	std::string flight1 = "G:\\Data\\09_EXPORT\\las - 230818_023505.las";
	std::string flight2 = "G:\\Data\\09_EXPORT\\las - 230818_024223.las";
	// 创建 PCL 点云对象
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
	double x_offset = 0, y_offset = 0, z_offset = 0; // 偏置，避免坐标数值过大
	read_las(flight1, cloud1, x_offset, y_offset, z_offset);
	read_las(flight2, cloud2, x_offset, y_offset, z_offset);

	Quality_Check(cloud1, cloud2);

	return 0;
}
