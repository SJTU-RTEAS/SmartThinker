#!/usr/bin/env python3
"""
简单的 Parquet 数据集信息查看工具
"""

import sys
import os


def get_parquet_info(file_path):
    """获取单个 Parquet 文件的详细信息"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("错误: 请先安装 pyarrow: pip install pyarrow")
        sys.exit(1)
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return None
    
    try:
        # 读取 Parquet 文件
        pf = pq.ParquetFile(file_path)
        
        # 获取元数据
        metadata = pf.metadata
        schema = pf.schema_arrow
        
        info = {
            'file_name': os.path.basename(file_path),
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'num_rows': metadata.num_rows,
            'num_row_groups': metadata.num_row_groups,
            'num_columns': metadata.num_columns,
            'created_by': metadata.created_by,
            'schema': schema,
            'column_names': [schema[i].name for i in range(len(schema))],
            'column_types': [str(schema[i].type) for i in range(len(schema))],
        }
        
        # 获取每列的统计信息（第一个 row group）
        if metadata.num_row_groups > 0:
            rg_meta = metadata.row_group(0)
            column_stats = []
            for i in range(rg_meta.num_columns):
                col = rg_meta.column(i)
                column_stats.append({
                    'path': col.path_in_schema,
                    'type': col.physical_type,
                    'encodings': col.encodings,
                    'total_uncompressed': col.total_uncompressed_size,
                    'total_compressed': col.total_compressed_size,
                })
            info['column_stats'] = column_stats
        
        return info
        
    except Exception as e:
        print(f"读取文件出错: {e}")
        return None


def print_file_info(info):
    """打印单个文件的信息"""
    if info is None:
        return
    
    print("=" * 60)
    print(f"📄 文件: {info['file_name']}")
    print("=" * 60)
    
    print(f"\n📊 基本信息:")
    print(f"  • 文件大小: {info['file_size_mb']:.2f} MB")
    print(f"  • 总行数: {info['num_rows']:,}")
    print(f"  • Row Groups: {info['num_row_groups']}")
    print(f"  • 列数: {info['num_columns']}")
    if info['created_by']:
        print(f"  • 创建工具: {info['created_by']}")
    
    print(f"\n📋 Schema 信息:")
    print(f"  {'列名':<30} {'数据类型':<20}")
    print(f"  {'-'*30} {'-'*20}")
    for name, dtype in zip(info['column_names'], info['column_types']):
        print(f"  {name:<30} {dtype:<20}")
    
    # 打印列统计信息
    if 'column_stats' in info:
        print(f"\n📈 存储统计 (第一个 Row Group):")
        print(f"  {'列名':<25} {'原始大小':>12} {'压缩后':>12} {'压缩率':>10}")
        print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
        for stat in info['column_stats']:
            orig = stat['total_uncompressed']
            comp = stat['total_compressed']
            ratio = (1 - comp/orig) * 100 if orig > 0 else 0
            print(f"  {stat['path']:<25} {orig:>12,} {comp:>12,} {ratio:>9.1f}%")
    
    print()


def scan_directory(dir_path):
    """扫描目录中的所有 Parquet 文件"""
    parquet_files = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.parquet'):
                parquet_files.append(os.path.join(root, f))
    return sorted(parquet_files)


def print_dataset_summary(file_paths):
    """打印数据集整体摘要"""
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa
    except ImportError:
        return
    
    total_rows = 0
    total_size = 0
    
    print("\n" + "=" * 60)
    print("📁 数据集整体摘要")
    print("=" * 60)
    
    # 尝试用数据集方式读取
    try:
        # 使用 PyArrow Dataset API（支持分区数据集）
        ds = pa.dataset.dataset(file_paths[0].replace(os.path.basename(file_paths[0]), ''))
        print(f"  • 检测到分区数据集: {type(ds).__name__}")
        print(f"  • 分区信息: {ds.partitioning if hasattr(ds, 'partitioning') else '无'}")
    except:
        pass
    
    # 统计所有文件
    all_schemas = []
    for fp in file_paths:
        try:
            pf = pq.ParquetFile(fp)
            total_rows += pf.metadata.num_rows
            total_size += os.path.getsize(fp)
            all_schemas.append(str(pf.schema_arrow))
        except:
            pass
    
    print(f"\n  • 总文件数: {len(file_paths)}")
    print(f"  • 总行数: {total_rows:,}")
    print(f"  • 总大小: {total_size / (1024*1024):.2f} MB")
    print(f"  • 平均每文件: {total_rows / len(file_paths):,.0f} 行")
    
    # 检查 Schema 一致性
    if len(set(all_schemas)) == 1:
        print(f"  • Schema 一致性: ✓ 所有文件 Schema 相同")
    else:
        print(f"  • Schema 一致性: ✗ 检测到 {len(set(all_schemas))} 种不同 Schema")
    
    print()


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python parquet_info.py <parquet文件或目录>")
        print("示例:")
        print("  python parquet_info.py data.parquet")
        print("  python parquet_info.py /path/to/parquet/dataset/")
        sys.exit(1)
    
    path = sys.argv[1]
    
    # 检查依赖
    try:
        import pyarrow
    except ImportError:
        print("请先安装 pyarrow: pip install pyarrow")
        sys.exit(1)
    
    if os.path.isfile(path):
        # 单个文件
        info = get_parquet_info(path)
        print_file_info(info)
        
    elif os.path.isdir(path):
        # 目录
        files = scan_directory(path)
        
        if not files:
            print(f"在 {path} 中未找到 Parquet 文件")
            sys.exit(1)
        
        print(f"在 {path} 中找到 {len(files)} 个 Parquet 文件")
        
        # 显示前5个文件的详细信息
        for i, fp in enumerate(files[:5], 1):
            print(f"\n[{i}/{min(5, len(files))}]")
            info = get_parquet_info(fp)
            print_file_info(info)
        
        if len(files) > 5:
            print(f"\n... 还有 {len(files)-5} 个文件未显示")
        
        # 打印整体摘要
        print_dataset_summary(files)
        
    else:
        print(f"路径不存在: {path}")


if __name__ == "__main__":
    main()