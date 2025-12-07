import os
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

def set_plot_chineese():
    if os.name == 'nt':  # Windows
        win_font_path = r"C:\Windows\Fonts\msjh.ttc"
        if os.path.exists(win_font_path):
            font_path = win_font_path
            print(f"Detected Windows — Using font: {font_path}")
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.family'] = 'Microsoft JhengHei'
        else:
            print("找不到 Windows 字體:C:\\Windows\\Fonts\\msjh.ttc,請確認是否存在")

    else:  # Linux/Mac
        font_path = "/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"
        plt.rcParams['font.family'] = 'WenQuanYi Zen Hei'

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
    else:
        print("Font file not found:", font_path)        