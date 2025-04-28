import os
import time
import json
import re
import sys
import requests
from you_get.common import any_download
from you_get.common import get_content
import game
import subprocess

def resource_path(relative_path):
    """ 通用资源路径解析 """
    try:
        # 打包后模式
        base_path = sys._MEIPASS
    except AttributeError:
        # 开发模式或独立exe模式
        if getattr(sys, 'frozen', False):
            # exe所在目录
            base_path = os.path.dirname(sys.executable)
        else:
            # 源码所在目录
            base_path = os.path.dirname(__file__)
    
    full_path = os.path.join(base_path, relative_path)
    return full_path

fake_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "zh-CN,zh;q=0.9",
    "Referer": "https://www.bilibili.com/"
}

def get_bilibili_cover(url):
    # 提取视频bvid（如从URL中解析）
    bvid = re.search(r'BV[\w]+', url).group()
    
    # 调用B站API
    api_url = f'https://api.bilibili.com/x/web-interface/view?bvid={bvid}'
    json_data = json.loads(get_content(api_url, headers=fake_headers))
    cover_url=json_data['data']['pic']  # 封面URL
    with requests.get(cover_url, stream=True, headers=fake_headers) as r:
            r.raise_for_status()
            with open(resource_path("bili_cover.jpg"), 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
            print(f"封面已保存为: bili_cover.jpg")
    return None
def download_bilibili_audio(url):
    
    # 临时下载目录
    temp_dir = resource_path("temp_download")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        
        # 强制指定音频格式并合并文件
        any_download(url,
                   output_dir=temp_dir, 
                   merge=False,  # 必须合并文件
                   format='dash-flv360-audio')
    except Exception as e:
        print(f"下载失败: {e}", flush=True)
        os.rmdir(temp_dir)
        return None
    
     # 修改后的文件筛选逻辑
    downloaded_files = sorted(
        [f for f in os.listdir(temp_dir) 
        if f.endswith('.mp4')]
    )

    # 新增清理[00].mp4文件的逻辑
    for f in downloaded_files:
        if f.endswith('[00].mp4'):
            target_file = os.path.join(temp_dir, f)
            os.remove(target_file)
            print(f"已删除分片文件: {f}", flush=True)

    # 重新获取有效文件列表（需排除已删除的[00].mp4）
    downloaded_files = sorted(
        [f for f in os.listdir(temp_dir) 
        if f.endswith('.mp4')]
    )
    
    if not downloaded_files:
        print("未找到下载文件", flush=True)
        return None
    
    # 新增重命名操作
    original_file = os.path.join(temp_dir, downloaded_files[0])
    renamed_file = os.path.join(temp_dir, "audio.mp4")
    
    try:
        if os.path.exists(renamed_file):
            os.remove(renamed_file)  # 删除已存在的旧文件
        os.rename(original_file, renamed_file)
    except Exception as e:
        print(f"文件重命名失败: {e}", flush=True)
        return None
    
    input_file = os.path.join(temp_dir, renamed_file)
    output_file = resource_path("bili_music.mp3")
    
    # 转换音频格式
    try:
        def get_ffmpeg_path():
            # 修正后的路径处理
            ffmpeg_dir1 = os.path.join("ffmpeg-master-latest-win64-gpl-shared", "ffmpeg-master-latest-win64-gpl-shared")
            ffmpeg_dir2 = os.path.join(ffmpeg_dir1, "bin")
            ffmpeg_exe = "ffmpeg.exe"
            
            # 构建完整路径
            local_path = resource_path(os.path.join(ffmpeg_dir2, ffmpeg_exe))
            
            # 调试输出
            print(f"检测本地ffmpeg路径: {local_path}", flush=True)
            
            if os.path.isfile(local_path):
                print(f"使用本地ffmpeg: {local_path}", flush=True)
                return local_path
            else:
                print("未找到本地ffmpeg，尝试环境变量", flush=True)
                return "ffmpeg"  # 回退到环境变量查找

        ffmpeg_path = resource_path(get_ffmpeg_path())
        print(f'"{ffmpeg_path}" -y -i "{input_file}" -vn -c:a libmp3lame -q:a 0 "{output_file}"', flush=True)
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", input_file,
            "-vn",
            "-c:a", "libmp3lame",
            "-q:a", "0",
            output_file
        ]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"转换失败: {e}", flush=True)
        os.remove(input_file)
        os.rmdir(temp_dir)
        return None
    
    # 清理临时文件
    try:
        os.remove(input_file)
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"清理失败: {e}", flush=True)
    
    return output_file


if os.path.exists(resource_path("bili_music.mp3")):
    i=input("当前目录已存在bili_music.mp3，是否覆盖？y/n : ")
    if i=="y":
        video_url = input("请输入B站视频链接：")
        audio_file = download_bilibili_audio(video_url)
        if audio_file:
            get_bilibili_cover(video_url)
            print(f"音频已保存为：{audio_file}", flush=True)
            time.sleep(1)
            print("加载游戏", flush=True)
            game = game.RhythmGame(resource_path("bili_music.mp3"))
            game.run()
        else:
            print("音频下载失败,找找自己的问题。", flush=True)
            time.sleep(1)
            print("退出", flush=True)
    else:
        print("加载游戏", flush=True)
        game = game.RhythmGame(resource_path("bili_music.mp3"))
        game.run()
        exit()
else:
    video_url = input("请输入B站视频链接：")
    audio_file = download_bilibili_audio(video_url)
    if audio_file:
        get_bilibili_cover(video_url)
        print(f"音频已保存为：{audio_file}", flush=True)
        time.sleep(1)
        print("加载游戏", flush=True)
        game = game.RhythmGame(resource_path("bili_music.mp3"))
        game.run()
        
    else:
        print("音频下载失败,找找自己的问题。", flush=True)
        time.sleep(1)
        print("退出", flush=True)        