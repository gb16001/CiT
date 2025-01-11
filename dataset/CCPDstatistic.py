'''
[=]calculate the centerpoint by bounding box
[=]statistic LP size by bbox
[=]statistic char size by LP size, inside structure of LP
img size=[720, 1160]
LP size=[240,116]
'''
import pandas as pd
import os
from PIL import Image, ImageDraw

def calcu_centerPoint(cvsFile: str):
    # 读取CSV文件
    df = pd.read_csv(cvsFile)

    # 提取相关列
    x1 = df["bounding_box_1_x"]
    y1 = df["bounding_box_1_y"]
    x2 = df["bounding_box_2_x"]
    y2 = df["bounding_box_2_y"]

    # 计算中心点
    df["center_x"] = (x1 + x2) / 2
    df["center_y"] = (y1 + y2) / 2

    # 保存为新的CSV文件
    base_name, ext = os.path.splitext(cvsFile)
    new_file_name = f"{base_name}_c{ext}"
    df.to_csv(new_file_name, index=False)

    return new_file_name

def statistic_LP_size(cvsFile: str):
    # 读取CSV文件
    df = pd.read_csv(cvsFile)
    
    # 提取相关列
    x1 = df["bounding_box_1_x"]
    y1 = df["bounding_box_1_y"]
    x2 = df["bounding_box_2_x"]
    y2 = df["bounding_box_2_y"]
    
    # 计算宽度和高度
    width = x2 - x1
    height = y2 - y1
    
    # 计算面积
    area = width * height
    
    # 添加到DataFrame中
    df['width'] = width
    df['height'] = height
    df['area'] = area
    
    
    # 输出分布列和平均值
    print("Bounding box dimensions (width and height), area:")
    print(df[['width', 'height', 'area']].describe().round(1))

    return df[['width', 'height', 'area']]  # 返回宽度、高度和面积列


class draw_LP_boxes:
    @staticmethod
    def draw_bbox_and_vertices(image_path, bbox, vertices, center, save_path):
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Define the dimensions of the rectangle
        rect_width = 240
        rect_height = 116

        # Calculate the top-left and bottom-right coordinates
        top_left = center[0] - rect_width // 2, center[1] - rect_height // 2
        bottom_right = center[0] + rect_width // 2, center[1] + rect_height // 2

        # Draw the rectangle centered on 'center'
        draw.rectangle([top_left, bottom_right], outline="yellow", width=3)
        # 绘制边界框
        draw.rectangle(bbox, outline="red", width=3)

        # 绘制顶点四边形
        draw.polygon(vertices, outline="blue", width=3)

        # 绘制中心点
        draw.ellipse([center[0]-5, center[1]-5, center[0]+5, center[1]+5], fill="green")
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir,exist_ok=True)
        # 保存图像
        image.save(save_path)
    @staticmethod
    def calculate_area(vertices):
        # 使用顶点四边形的坐标计算面积
        x1, y1, x2, y2, x3, y3, x4, y4 = vertices
        area = 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - (y1*x2 + y2*x3 + y3*x4 + y4*x1))
        return area
    @staticmethod
    def char_in_LP(cvsFile: str,basePath:str):
        # 读取CSV文件
        df = pd.read_csv(cvsFile)
        
        # 随机抽取5个样本
        samples = df.sample(n=5)

        results = []

        for index, row in samples.iterrows():
            filename = row['filename']
            image_path = os.path.join(basePath,row['CCPD_path'], filename)
            bbox = [(row['bounding_box_1_x'], row['bounding_box_1_y']), 
                    (row['bounding_box_2_x'], row['bounding_box_2_y'])]
            vertices = [(row['vertex_1_x'], row['vertex_1_y']), 
                        (row['vertex_2_x'], row['vertex_2_y']), 
                        (row['vertex_3_x'], row['vertex_3_y']), 
                        (row['vertex_4_x'], row['vertex_4_y'])]
            center = (row['center_x'], row['center_y'])

            # 计算面积
            bbox_area = (bbox[1][0] - bbox[0][0]) * (bbox[0][1] - bbox[1][1])
            vertices_area = draw_LP_boxes.calculate_area([v for vertex in vertices for v in vertex])
            area_ratio = abs(vertices_area / bbox_area) 

            # 保存带标记的图像
            save_path = f"./imgs/{filename}"
            
            draw_LP_boxes.draw_bbox_and_vertices(image_path, bbox, vertices, center, save_path)

            # 记录结果
            results.append((filename, image_path, area_ratio))

        # 返回处理结果
        return results
pass


if __name__ == "__main__":
    csvFile="dataset/CCPDanno_test.csv"
    calcu_centerPoint(csvFile)
    # statistic_LP_size(csvFile)
    results = draw_LP_boxes.char_in_LP(csvFile,'dataset')
    for r in results:
        print(r ) 
    pass
