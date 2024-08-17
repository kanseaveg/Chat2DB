import os
import sqlite3
import shutil

def get_all_databases(base_path):
    # 获取所有数据库文件的路径
    databases = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.sqlite'):
                databases.append(os.path.join(root, file))
    return databases

def get_all_tables(conn):
    # 获取数据库中所有表的名称
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def is_table_empty(conn, table_name):
    # 检查表是否为空
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
    count = cursor.fetchone()[0]
    return count == 0

def delete_database_and_folder(db_path):
    # 删除数据库文件及其文件夹
    dir_path = os.path.dirname(db_path)
    shutil.rmtree(dir_path)
    print(f"Deleted database and folder: {dir_path}")

def process_database(db_path, deleted_databases):
    conn = sqlite3.connect(db_path)
    tables = get_all_tables(conn)
    
    for table in tables:
        if is_table_empty(conn, table):
            conn.close()
            delete_database_and_folder(db_path)
            deleted_databases.append(db_path)
            return  # 只要发现一个空表，就删除数据库并退出处理

    conn.close()

def main():
    base_path = 'database'
    databases = get_all_databases(base_path)
    
    deleted_databases = []
    
    for db_path in databases:
        process_database(db_path, deleted_databases)
    
    # 汇总输出删除信息
    print("\nSummary:")
    print("Deleted databases:")
    for db_path in deleted_databases:
        print(f"Database: {db_path}")

if __name__ == "__main__":
    main()
