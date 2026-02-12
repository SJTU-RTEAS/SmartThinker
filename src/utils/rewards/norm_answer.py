import re

def normalize_latex(text: str) -> str:
    """规范化 LaTeX 表达式，统一格式变体"""
    if not text:
        return text
    
    # 1. 统一分数格式
    text = text.replace('\\dfrac', '\\frac')
    text = text.replace('\\tfrac', '\\frac')
    text = text.replace('\\cfrac', '\\frac')
    
    # 2. 移除显示样式命令
    text = text.replace('\\displaystyle', '')
    text = text.replace('\\textstyle', '')
    text = text.replace('\\scriptstyle', '')
    
    # 3. 统一根号格式
    text = text.replace('\\sqrt', '\\sqrt')  # 保持不变，但可以处理变体
    
    # 4. 统一括号格式
    text = text.replace('\\left(', '(')
    text = text.replace('\\right)', ')')
    text = text.replace('\\left[', '[')
    text = text.replace('\\right]', ']')
    text = text.replace('\\left{', '{')
    text = text.replace('\\right}', '}')
    text = text.replace('\\bigl(', '(')
    text = text.replace('\\bigr)', ')')
    text = text.replace('\\Bigl(', '(')
    text = text.replace('\\Bigr)', ')')
    
    # 5. 移除多余空格和格式字符
    text = text.replace('\\,', '')
    text = text.replace('\\;', '')
    text = text.replace('\\:', '')
    text = text.replace('\\ ', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text