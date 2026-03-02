import os
from PIL import Image
import sys

def split_to_3x3(image_path: str, out_dir: str = "."):
    img = Image.open(image_path)
    w, h = img.size

    # 九宫格：每格的“基础宽高”
    tile_w = w // 3
    tile_h = h // 3

    base = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    idx = 1
    for r in range(3):          # 上到下
        for c in range(3):      # 左到右
            left = c * tile_w
            upper = r * tile_h
            # 最后一列/行把余数吃掉，避免丢像素
            right = (c + 1) * tile_w if c < 2 else w
            lower = (r + 1) * tile_h if r < 2 else h

            tile = img.crop((left, upper, right, lower))
            out_path = os.path.join(out_dir, f"{base}_{idx:02d}.png")
            tile.save(out_path)
            print("saved:", out_path)
            idx += 1

if __name__ == "__main__":
    # 优先使用命令行参数：
    #   1) 第一个参数: 图片路径（绝对或相对）
    #   2) 第二个参数: 输出目录（可选，默认为脚本目录）
    script_dir = os.path.dirname(__file__)
    default_img = os.path.join(script_dir, "火焰海滩-传输前.png")
    img_path = sys.argv[1] if len(sys.argv) > 1 else default_img
    out_dir = sys.argv[2] if len(sys.argv) > 2 else script_dir

    # 如果给的是相对路径，先尝试当前工作目录，再尝试脚本目录
    if not os.path.isabs(img_path):
        if not os.path.exists(img_path):
            alt = os.path.join(script_dir, img_path)
            if os.path.exists(alt):
                img_path = alt

    if not os.path.exists(img_path):
        print("错误：找不到文件 ->", img_path)
        print("脚本目录 (%s) 下的文件列表：" % script_dir)
        for f in os.listdir(script_dir):
            print("  ", f)
        sys.exit(1)

    split_to_3x3(img_path, out_dir=out_dir)
