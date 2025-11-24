"""
该脚本用于将 ZEDX（HTML 文档 + 目录树）数据集转换为便于检索的纯文本与映射文件：
- 遍历 `nodetree.xml` 目录树，定位每个 HTML 文档的路径
- 解析 HTML：
  - 为 `span.xref gxref` 的术语补充英文/中文缩略解释
  - 抽取 `figure` 的图片标题与说明，并复制图片目录到目标数据目录
  - 将 HTML 转换为纯文本（去链接/去图片），并可在文本前追加“文档路径”
- 生成两类映射：
  - `imgmap_raw.json`：文本文件路径 -> {图片标题 -> 图片路径与标题说明}
  - `pathmap.json`：文本文件路径 -> 知识树路径（用于后续按路径检索）

脚本依赖于以下全局变量（在 main 中初始化）：
- `base_meta_dir`：原始数据目录（含 documents 与 nodetree.xml）
- `base_processed_data_dir`：处理后的输出目录（文本与图片）
- `package_list`：要处理的包名称列表
- `args.with_path`：是否在转换后的文本前插入“文档路径”头部
- `filepath2imgpath_dict`：累积的图片映射字典
- `filepath_2_knowpath`：累积的路径映射字典
"""

# 标准库：JSON 读写，用于输出映射文件
import json
# 标准库：文件/路径操作
import os
# 标准库：目录复制（用于复制 images 子目录）
import shutil
# 标准库：解析 XML 目录树（nodetree.xml）
from xml.etree import ElementTree
# 第三方：解析 HTML 文档结构
from bs4 import BeautifulSoup
# 第三方：进度条展示
from tqdm import tqdm
# 第三方：将 HTML 转换为纯文本
import html2text
# 标准库：URL 解码（处理图片 src 包含的转义）
import urllib.parse


def dfs_tree(url2path: dict, node, parents: tuple):
    """深度优先遍历 nodetree.xml，将每个文档 URL 映射到其知识树路径。

    参数：
    - url2path：累积映射 {url -> 路径元组}
    - node：当前 XML 节点
    - parents：当前节点在知识树中的父路径（元组）
    """
    for child in node:  # 遍历当前节点的所有子节点
        sub_parents = parents + (child.get('name'),)  # 追加子节点名称形成子路径
        url = child.get('url')  # 读取子节点的文档 URL
        url = url.replace('\\', '/')  # 将 Windows 风格分隔符统一为 POSIX 风格
        url2path[url] = sub_parents  # 记录 URL 到知识树路径的映射
        dfs_tree(url2path, child, sub_parents)  # 递归遍历子树


def process_hmtl(html_doc, meta_dir, build_data_dir, url):
    """解析 HTML 文档，补充缩略语、抽取图片信息，并转为纯文本。

    说明：使用到的 `package_name` 与 `filepath2imgpath_dict` 为 main 中的全局变量。
    """
    soup = BeautifulSoup(html_doc, "html.parser")  # 将 HTML 字符串解析为 DOM 树

    # 遍历所有缩略语节点，补充 title 中的英文/中文解释到可读文本
    for span in soup.find_all("span", class_="xref gxref"):
        title = span.get("title")  # 读取缩略语的 title 属性（包含“英文--中文”）
        if title:  # 如果存在 title，则尝试拆分并注入到文本中
            try:
                en, cn = title.split("--")  # 标准格式：英文--中文
                span.string = f"{span.string}({en}, {cn})"  # 在原术语后追加括注
            except:
                span.string = f"{span.string}({title})"  # 兜底：无法拆分时直接追加 title
                print('error')  # 打印错误提示，便于定位异常格式

    # 遍历文档中的所有 figure，抽取图片路径与标题说明，并复制图片目录
    for figure in soup.find_all("figure", class_="fig fig_ fignone"):
        figure_title = figure.find('span').string  # 图片标题在第一个 span 元素中
        if not figure_title.startswith('图'):  # 非规范标题时打印提示
            print('没有图片标题:', figure)
        else:
            figure_title = figure_title.strip()  # 去除首尾空白字符
            try:
                figure_path = figure.find('img')['src']  # 读取图片相对路径
                try:
                    figure_path = urllib.parse.unquote(figure_path)  # 对 URL 进行解码（处理转义）
                except:
                    import traceback  # 引入 traceback 以便打印完整异常栈
                    traceback.print_exc()  # 打印异常详情，便于调试
            except:
                continue  # 如果 figure 中没有 img，则跳过该条

            dir_path = os.path.dirname(url)  # 当前文档的目录路径（相对）
            ori_img_dir = os.path.join(meta_dir, 'documents', dir_path, 'images')  # 原始 images 目录
            img_path = os.path.join(package_name, dir_path, figure_path)  # 输出映射中记录的图片路径（相对包名）
            # 在图片映射中为当前文本文件创建记录，并写入图片路径
            filepath2imgpath_dict.setdefault(
                os.path.join(package_name, url.replace('.html', '.txt').replace('.htm', '.txt')),
                {}
            )[figure_title] = {"img_path": img_path}

            img_dir = os.path.join(build_data_dir, dir_path, 'images')  # 目标 images 目录（用于复制）
            if not os.path.exists(img_dir) and os.path.exists(ori_img_dir):  # 若目标不存在且原始存在
                shutil.copytree(ori_img_dir, img_dir)  # 复制整个 images 目录到目标结构

            # 从 figcaption 中提取图片说明文本，并移除内嵌的 span 文本，得到纯净标题
            figure_cap = figure.find("figcaption")  # 图片说明容器
            all_text = figure_cap.get_text(separator=' ', strip=True)  # 提取整段说明文本
            span_texts = [span.get_text(separator=' ', strip=True) for span in figure_cap.find_all('span')]  # 内部 span 文本列表
            for span_text in span_texts:
                all_text = all_text.replace(span_text, '')  # 移除内嵌 span 文本，保留说明主体
            title = ' '.join(all_text.split())  # 规范化空白：多个空格折叠为单个空格
            # 在图片映射中写入该标题说明
            filepath2imgpath_dict[
                os.path.join(package_name, url.replace('.html', '.txt').replace('.htm', '.txt'))
            ][figure_title]["title"] = title

    html = str(soup)  # 将修改后的 DOM 树转回 HTML 字符串
    h = html2text.HTML2Text()  # 创建 HTML->Text 转换器
    h.ignore_links = True  # 忽略链接（不保留 URL）
    h.ignore_images = True  # 忽略图片（不保留图片占位）
    h.body_width = 0  # 不进行行宽限制（保留原始换行）
    text = h.handle(html)  # 执行转换得到纯文本

    return text  # 返回纯文本内容


def load_content(meta_dir, build_data_dir, url):
    """读取指定 URL 对应的 HTML 文件并进行转换；支持不同编码容错。"""
    load_path = os.path.join(meta_dir, 'documents', url)  # 构建源 HTML 的真实路径
    if os.path.exists(load_path):  # 如果文件存在则尝试读取
        try:
            html_doc = open(load_path, 'r', encoding='utf-8').read()  # 优先按 UTF-8 读取
        except:
            html_doc = open(load_path, 'r', encoding='gb2312').read()  # 兜底：按 GB2312 读取
        # 保留备选的“直接抽取所有文本”的方案（注释示例）：
        # soup = BeautifulSoup(html_doc, 'lxml')
        # texts = soup.find_all(string=True)
        # all_text = '\n'.join(texts)
        # return all_text
        return process_hmtl(html_doc, meta_dir, build_data_dir, url)  # 正常流程：走 HTML 解析与转换
    else:
        print('文档不存在: ', load_path)  # 文件缺失时提示
        return None  # 返回空，调用方需跳过


def format_content(content, path):
    """规范化纯文本内容并可选择在顶部增加“文档路径”头部。"""
    new_content = []  # 规范化后的行列表
    last_line = None  # 记录上一行，便于去除连续重复行
    for line in content.split('\n'):  # 按行遍历纯文本
        if last_line == line:  # 跳过与上一行相同的重复行
            continue
        last_line = line  # 更新上一行记录
        line = line.strip()  # 去除首尾空白
        if line.startswith('html'):  # 过滤以 html 开头的无意义行（转换器残留）
            print('start with html')
            continue
        if line:  # 保留非空行
            new_content.append(line)
    new_str = ''  # 初始化最终字符串
    if args.with_path:  # 根据命令行参数决定是否添加路径头部
        new_str += '###\n'  # 分隔头
        new_str += '文档路径: ' + '/'.join(path) + '\n\n'  # 以“/”拼接知识树路径

    if new_content:  # 若存在有效内容，拼接为文本块
        new_str += '\n'.join(new_content) + '\n'
    else:  # 否则提示为空并写入占位
        print('空文档: ', path)
        new_str += '<文档为空>\n'
    return new_str  # 返回规范化文本


def fill_document(meta_dir, build_data_dir, url2path):
    """遍历 URL->路径 映射，生成对应的 .txt 文本与路径映射。"""
    filepath_2_knowpath_ = dict()  # 局部路径映射累积（最终合并到全局）
    for url, path in tqdm(url2path.items()):  # 逐条处理每个文档 URL
        content = load_content(meta_dir, build_data_dir, url)  # 加载并转换 HTML 文本
        if content is None:  # 缺失或异常则跳过
            continue
        if url.endswith('.html') or url.endswith('.htm'):  # 仅处理 HTML/HTM 后缀
            url = url.replace('.html', '.txt').replace('.htm', '.txt')  # 输出改为 .txt 后缀
            build_file = os.path.join(build_data_dir, url)  # 目标文本文件路径
            build_dir = os.path.dirname(build_file)  # 目标目录路径
            os.makedirs(build_dir, exist_ok=True)  # 确保目标目录存在
            filepath_2_knowpath_["/".join([path[0], url])] = path  # 记录文件路径到知识树路径的映射
            with open(build_file, 'w', encoding='utf-8') as fin:  # 写入规范化后的文本内容
                fin.write(format_content(content, path))
        else:
            print('未知的url后缀: ', url)  # 遇到无法识别的后缀时提示
    return filepath_2_knowpath_  # 返回局部映射以便上层合并


def process_package(package_name):
    """处理单个包：解析目录树、转换所有文档并生成映射。"""
    meta_dir = os.path.join(base_meta_dir, package_name)  # 原始数据包根目录
    build_data_dir = os.path.join(base_processed_data_dir, package_name)  # 输出数据包根目录
    os.makedirs(build_data_dir, exist_ok=True)  # 保证输出目录存在

    node_tree_path = os.path.join(meta_dir, 'nodetree.xml')  # 目录树 XML 文件路径

    element_tree = ElementTree.fromstring(open(node_tree_path, 'r', encoding='utf-8').read())  # 读取并解析 XML

    url2path = {}  # 初始化 URL->路径 映射字典
    dfs_tree(url2path, element_tree, (package_name,))  # 从根节点开始遍历，填充映射
    filepath_2_knowpath_ = fill_document(meta_dir, build_data_dir, url2path)  # 生成文本与路径映射
    filepath_2_knowpath.update(filepath_2_knowpath_)  # 合并到全局路径映射


if __name__ == '__main__':
    import argparse  # 标准库：命令行参数解析

    parse = argparse.ArgumentParser()  # 创建参数解析器
    parse.add_argument('--with_path', action='store_true', default=False)  # 是否在文本前输出“文档路径”头部
    args = parse.parse_args()  # 解析命令行参数
    print('with_path: ', args.with_path)  # 回显参数设置，便于确认运行模式
    # exit()  # 调试时可提前退出（此处保留注释以示用途）

    base_meta_dir = 'src/data/origin_data'  # 原始数据根目录（包含每个包的 documents 与 nodetree.xml）
    base_processed_data_dir = 'src/data/format_data_with_img'  # 处理后的输出根目录（文本与图片复制）
    filepath_2_knowpath = dict()  # 全局：文件路径 -> 知识树路径 映射
    filepath2imgpath_dict = dict()  # 全局：文件路径 -> {图片标题 -> 图片信息} 映射
    package_list = ['director', 'emsplus', 'rcp', 'umac']  # 待处理的包名称列表
    for package_name in package_list:  # 逐包处理
        process_package(package_name)  # 执行包级处理：解析目录树、转换文档并累积映射
    with open(os.path.join(base_processed_data_dir, "imgmap_raw.json"), 'w') as f:  # 输出图片映射到 JSON 文件
        f.write(json.dumps(filepath2imgpath_dict, ensure_ascii=False, indent=4))  # 使用中文友好与缩进格式
    with open(os.path.join(base_processed_data_dir, "pathmap.json"), 'w') as f:  # 输出路径映射到 JSON 文件
        f.write(json.dumps(filepath_2_knowpath, ensure_ascii=False, indent=4))  # 使用中文友好与缩进格式
