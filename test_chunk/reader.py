import os
from pathlib import Path
from typing import Union, List, Dict
import mimetypes


class FileReader:
    """多格式文件读取器，支持 txt, pdf, docx, html, md, json, csv, xml 等"""

    def __init__(self, encoding: str = 'utf-8', errors: str = 'ignore'):
        """
        初始化文件读取器
        
        Args:
            encoding: 文件编码，默认 utf-8
            errors: 编码错误处理方式，默认 ignore（忽略错误）
        """
        self.encoding = encoding
        self.errors = errors

    def _read_pdf(self, file_path: str) -> str:
        """读取 PDF 文件"""
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = []
                for page in reader.pages:
                    text.append(page.extract_text())
                return '\n'.join(text)
        except ImportError:
            return "[错误: 需要安装 PyPDF2 库: pip install PyPDF2]"
        except Exception as e:
            return f"[PDF读取失败: {str(e)}]"

    def _read_docx(self, file_path: str) -> str:
        """读取 Word 文档 (.docx)"""
        try:
            import docx
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return '\n'.join(text)
        except ImportError:
            return "[错误: 需要安装 python-docx 库: pip install python-docx]"
        except Exception as e:
            return f"[DOCX读取失败: {str(e)}]"

    def _read_doc(self, file_path: str) -> str:
        """读取旧版 Word 文档 (.doc)"""
        try:
            import subprocess
            # 尝试使用 antiword (Linux) 或 textutil (Mac)
            try:
                result = subprocess.run(['antiword', file_path],
                                        capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout
            except FileNotFoundError:
                pass

            return "[错误: .doc 格式需要安装 antiword 工具或转换为 .docx]"
        except Exception as e:
            return f"[DOC读取失败: {str(e)}]"

    def _read_html(self, file_path: str) -> str:
        """读取 HTML 文件并提取文本"""
        try:
            from bs4 import BeautifulSoup
            with open(file_path, 'r', encoding=self.encoding, errors=self.errors) as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator='\n', strip=True)
        except ImportError:
            return "[错误: 需要安装 beautifulsoup4 库: pip install beautifulsoup4]"
        except Exception as e:
            return f"[HTML读取失败: {str(e)}]"

    def _read_csv(self, file_path: str) -> str:
        """读取 CSV 文件"""
        try:
            import csv
            with open(file_path, 'r', encoding=self.encoding, errors=self.errors) as f:
                reader = csv.reader(f)
                rows = []
                for row in reader:
                    rows.append(' | '.join(row))
                return '\n'.join(rows)
        except Exception as e:
            return f"[CSV读取失败: {str(e)}]"

    def _read_excel(self, file_path: str) -> str:
        """读取 Excel 文件 (.xlsx, .xls)"""
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            text = []
            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text.append(f"\n=== {sheet_name} ===\n")
                for row in sheet.iter_rows(values_only=True):
                    row_text = ' | '.join([str(cell) if cell is not None else '' for cell in row])
                    if row_text.strip():
                        text.append(row_text)
            return '\n'.join(text)
        except ImportError:
            return "[错误: 需要安装 openpyxl 库: pip install openpyxl]"
        except Exception as e:
            return f"[Excel读取失败: {str(e)}]"

    def _read_json(self, file_path: str) -> str:
        """读取 JSON 文件"""
        try:
            import json
            with open(file_path, 'r', encoding=self.encoding, errors=self.errors) as f:
                data = json.load(f)
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"[JSON读取失败: {str(e)}]"

    def _read_xml(self, file_path: str) -> str:
        """读取 XML 文件"""
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(file_path)
            root = tree.getroot()

            def extract_text(element, level=0):
                text = []
                indent = '  ' * level
                if element.text and element.text.strip():
                    text.append(f"{indent}{element.tag}: {element.text.strip()}")
                else:
                    text.append(f"{indent}{element.tag}")
                for child in element:
                    text.extend(extract_text(child, level + 1))
                return text

            return '\n'.join(extract_text(root))
        except Exception as e:
            return f"[XML读取失败: {str(e)}]"

    def _read_markdown(self, file_path: str) -> str:
        """读取 Markdown 文件"""
        try:
            with open(file_path, 'r', encoding=self.encoding, errors=self.errors) as f:
                return f.read()
        except Exception as e:
            return f"[Markdown读取失败: {str(e)}]"

    def _read_text(self, file_path: str) -> str:
        """读取纯文本文件"""
        try:
            with open(file_path, 'r', encoding=self.encoding, errors=self.errors) as f:
                return f.read()
        except Exception as e:
            return f"[文本读取失败: {str(e)}]"

    def read_file(self, file_path: str) -> str:
        """
        自动识别文件格式并读取
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串
        """
        if not os.path.exists(file_path):
            return f"[错误: 文件不存在 - {file_path}]"

        ext = os.path.splitext(file_path)[1].lower()

        # 根据扩展名选择读取方法
        readers = {
            '.pdf': self._read_pdf,
            '.docx': self._read_docx,
            '.doc': self._read_doc,
            '.html': self._read_html,
            '.htm': self._read_html,
            '.csv': self._read_csv,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel,
            '.json': self._read_json,
            '.xml': self._read_xml,
            '.md': self._read_markdown,
            '.markdown': self._read_markdown,
            '.txt': self._read_text,
            '.log': self._read_text,
            '.py': self._read_text,
            '.js': self._read_text,
            '.java': self._read_text,
            '.cpp': self._read_text,
            '.c': self._read_text,
            '.sh': self._read_text,
            '.yaml': self._read_text,
            '.yml': self._read_text,
            '.ini': self._read_text,
            '.cfg': self._read_text,
        }

        reader_func = readers.get(ext, self._read_text)
        return reader_func(file_path)

    def read_files(self, file_paths: List[str], return_dict: bool = True) -> Union[Dict[str, str], List[str]]:
        """
        读取多个文件
        
        Args:
            file_paths: 文件路径列表
            return_dict: 是否返回字典格式（文件名:内容），False则返回内容列表
            
        Returns:
            字典或列表格式的文件内容
        """
        if return_dict:
            result = {}
            for path in file_paths:
                filename = os.path.basename(path)
                result[filename] = self.read_file(path)
            return result
        else:
            return [self.read_file(path) for path in file_paths]

    def read_directory(self,
                       dir_path: str,
                       extensions: List[str] = None,
                       recursive: bool = False,
                       return_dict: bool = True) -> Union[Dict[str, str], str]:
        """
        读取目录下的所有文件
        
        Args:
            dir_path: 目录路径
            extensions: 文件扩展名过滤列表，如 ['.txt', '.pdf']，None则读取所有支持的文件
            recursive: 是否递归读取子目录
            return_dict: 是否返回字典格式（文件名:内容）
            
        Returns:
            字典格式的文件内容或拼接的字符串
        """
        result = {}

        if not os.path.isdir(dir_path):
            return {"错误": f"目录不存在: {dir_path}"}

        # 获取文件列表
        if recursive:
            file_list = []
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_list.append(os.path.join(root, file))
        else:
            file_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
                         if os.path.isfile(os.path.join(dir_path, f))]

        # 过滤扩展名
        if extensions:
            extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}'
                          for ext in extensions]
            file_list = [f for f in file_list
                         if os.path.splitext(f)[1].lower() in extensions]

        # 读取文件
        for file_path in file_list:
            relative_path = os.path.relpath(file_path, dir_path)
            result[relative_path] = self.read_file(file_path)

        if return_dict:
            return result
        else:
            # 拼接成字符串
            output = []
            for filename, content in result.items():
                output.append(f"{'=' * 60}")
                output.append(f"文件: {filename}")
                output.append(f"{'=' * 60}")
                output.append(content)
                output.append("\n")
            return "\n".join(output)

    def get_supported_formats(self) -> List[str]:
        """返回支持的文件格式列表"""
        return [
            '.pdf', '.docx', '.doc',  # 文档
            '.html', '.htm',  # 网页
            '.csv', '.xlsx', '.xls',  # 表格
            '.json', '.xml',  # 数据格式
            '.md', '.markdown',  # Markdown
            '.txt', '.log',  # 纯文本
            '.py', '.js', '.java', '.cpp', '.c', '.sh',  # 代码
            '.yaml', '.yml', '.ini', '.cfg'  # 配置文件
        ]


# 使用示例
if __name__ == "__main__":
    reader = FileReader()

    print("=== 支持的文件格式 ===")
    print(", ".join(reader.get_supported_formats()))

    print("\n=== 示例1: 读取 PDF 文件 ===")
    pdf_content = reader.read_file("document.pdf")
    print(pdf_content[:200])

    print("\n=== 示例2: 读取 Word 文档 ===")
    docx_content = reader.read_file("report.docx")
    print(docx_content[:200])

    print("\n=== 示例3: 读取 HTML 文件 ===")
    html_content = reader.read_file("page.html")
    print(html_content[:200])

    print("\n=== 示例4: 读取多种格式文件 ===")
    files = ["data.json", "config.xml", "readme.md", "report.pdf"]
    results = reader.read_files(files, return_dict=True)
    for filename, content in results.items():
        print(f"\n{filename}:\n{content[:150]}...")

    print("\n=== 示例5: 读取目录（所有文档类型）===")
    docs = reader.read_directory(
        dir_path="./documents",
        extensions=['.pdf', '.docx', '.txt', '.md'],
        recursive=True
    )
    print(f"找到 {len(docs)} 个文档")

    print("\n=== 示例6: 读取 Excel 文件 ===")
    excel_content = reader.read_file("data.xlsx")
    print(excel_content[:300])
