# 导入必要的库
import librosa  # 音频处理库，用于分析音乐特征
import numpy as np  # 数学计算库
import pygame  # 游戏开发框架
import sys  # 系统相关功能
import random  # 随机数生成
import os
import math
from collections import deque  # 双端队列数据结构

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = BASE_DIR

    return os.path.join(base_path, relative_path)

class AudioAnalyzer:
    """增强版音频分析器，集成静音检测和智能事件过滤"""
    def __init__(self, file_path):
        # 先加载音频文件以获取时长
        self.duration = librosa.get_duration(path=file_path)
        
        # STEP 2: 动态选择采样率
        if self.duration > 300:  # 超过5分钟（300秒）
            sr = None           # 保留原始采样率
            print("警告：长音频启用原始采样率模式")
        else:
            sr = 44100          # 短音频使用标准采样率
        
        # STEP 3: 加载音频（根据时长选择不同模式）
        self.y, self.sr = librosa.load(file_path, sr=sr)
        # 预计算静音区间
        self.silence_intervals = self.detect_silence()

    def detect_silence(self, threshold_db=-15, min_silence_duration=0.2):
        """
        检测静音区间
        参数：
        threshold_db: 静音判定阈值（分贝），默认-40dB
        min_silence_duration: 最小静音持续时间（秒），默认0.5秒
        返回：静音区间列表[(start1, end1), ...]（单位：秒）
        """
        # 使用librosa静音分割算法
        non_silent_intervals = librosa.effects.split(
            self.y,
            top_db=abs(threshold_db),  # 转换为正数
            frame_length=2048,         # 增大帧长度提高低频敏感度
            hop_length=512             # 固定跳跃步长
        )
        
        # 转换为秒单位并反转得到静音区间
        silence = []
        prev_end = 0
        for (start, end) in non_silent_intervals:
            if start > prev_end:
                silence.append((prev_end/self.sr, start/self.sr))
            prev_end = end
        # 添加最后的静音段
        if prev_end < len(self.y):
            silence.append((prev_end/self.sr, len(self.y)/self.sr))
        
        # 过滤短时静音
        return [(s,e) for (s,e) in silence if (e-s) >= min_silence_duration]

    def is_in_silence(self, t):
        """检查时间点是否位于静音区间"""
        return any(start < t < end for (start, end) in self.silence_intervals)

    def extract_features(self):
        """智能特征提取，集成能量过滤和节拍修正"""
        # 计算RMS能量（每帧的均方根能量）
        rms = librosa.feature.rms(y=self.y, frame_length=128, hop_length=16)[0]
        
        # 三级阈值
        strong_threshold = np.percentile(rms, 80)  # 强音头阈值
        medium_threshold = np.percentile(rms, 35)  # 中音头阈值
        weak_threshold = np.percentile(rms, 1)   # 弱音头阈值 
    
        
        onset_env = librosa.onset.onset_strength(
            y=self.y, 
            sr=self.sr,
            hop_length=16,
            aggregate=np.median  # 使用中位数聚合
        )
        
        onset_env = onset_env / np.max(onset_env)
    
        # 应用三级阈值
        onset_env_strong = (onset_env >= strong_threshold).astype(float) * 0.25  # 强音头权重更高
        onset_env_medium = ((onset_env >= medium_threshold) & (onset_env < strong_threshold)).astype(float) * 0.4  # 中音头权重适中
        onset_env_weak = ((onset_env >= weak_threshold) & (onset_env < medium_threshold)).astype(float) * 0.75  # 弱音头权重较低
        onset_env_final = onset_env_strong + onset_env_medium + onset_env_weak
        onset_env_final = librosa.util.normalize(onset_env_final)
        
        # 节拍检测（使用修正版算法）
        tempo, beat_frames = librosa.beat.beat_track(
            y=self.y, sr=self.sr,
            units='frames',
            tightness=0.00000000001  # 提高节拍间隔一致性
        )
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)

        
        # 音头检测（带能量过滤）
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env_final,  # 特征包络线
            sr=self.sr,                     # 采样率
            units='frames',                 # 返回单位
            pre_max=1,                      # 前向最大检测窗口
            post_max=1,                     # 后向最大检测窗口
            pre_avg=1,                      # 前向平均窗口
            post_avg=1,                     # 后向平均窗口
            delta=0.00000000001,                     # 差异阈值
            wait=0                          # 最小间隔帧数
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        # 合并事件并过滤静音区
        all_events = np.union1d(beat_times, onset_times)
        filtered_events = [t for t in all_events if not self.is_in_silence(t)]
        
        # 时间点密度控制（防止过密）
        base_interval = 60 / tempo
        min_interval = base_interval * 0.06
        return self._filter_dense_events(filtered_events, min_interval), tempo

    def _filter_dense_events(self, events, min_interval):
        """事件密度控制，确保最小时间间隔"""
        filtered = []
        prev_t = -min_interval  # 初始化保证第一个事件通过
        for t in sorted(events):
            if t - prev_t >= min_interval:
                filtered.append(t)
                prev_t = t
        return filtered

class Note(pygame.sprite.Sprite):
    """音符游戏对象"""
    NOTE_WIDTH = 160  # 音符显示宽度（像素）
    LANE_OFFSET = 250  # 第一条轨道的X轴起始偏移量
    TRAVEL_DISTANCE = 620  # 音符移动总距离（从生成到判定线）
    
    def __init__(self, lane, note_type, speed, spawn_time):
        """
        参数：
        lane: 轨道编号（0-3）
        note_type: 音符类型（tap/click点击型，hold长按型）
        speed: 下落速度（像素/帧）
        spawn_time: 生成时间戳（毫秒）
        """
        super().__init__()
        self.fix=0
        self.lane = lane
        self.note_type = note_type  # 记录音符类型
        colors = {"tap": (0,255,0), "hold": (255,0,0)}  # 不同类型颜色定义
        height = 30 if note_type == "hold" else 30  # 长按型高度较小
        
        # 创建音符表面对象
        self.image = pygame.Surface((self.NOTE_WIDTH-40, height))
        self.image.fill(colors[note_type])  # 填充颜色
        self.rect = self.image.get_rect()  # 获取矩形碰撞区域
        
        # 设置初始位置：X轴根据轨道计算，Y轴在屏幕上方
        self.rect.centerx = self.LANE_OFFSET + lane * 250
        self.rect.top = -35
        
        if isinstance(speed, np.ndarray):
            speed = speed.item()
        self.speed = speed  # 下落速度
        self.spawn_time = spawn_time  # 生成时间戳
        self.expected_time = self.calculate_expected_time()  # 预期到达判定线时间

    def calculate_expected_time(self):
        """计算音符应到达判定线的精确时间"""
        # 公式：生成时间 + (移动距离/速度) * 每帧时间（ms）
        return self.spawn_time + (self.TRAVEL_DISTANCE / self.speed) * (1000/60)

    def update(self):
        """每帧更新位置（自动被Sprite组调用）"""
        self.rect.y += self.speed  # 垂直下落

class Particle(pygame.sprite.Sprite):
    """粒子效果"""
    def __init__(self, position,level):
        super().__init__()
        # 随机粒子参数
        self.size = random.randint(1, 5)  # 粒子尺寸
        angle = random.uniform(0, 2*math.pi)  # 随机运动角度
        self.speed = random.uniform(1.5, 15)  # 随机速度
        self.lifetime = random.randint(35, 250)  # 随机存活时间
        
        # 创建带透明度的表面
        self.image = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        # 随机暖色调（红/橙/黄）
        if level == 50:
            self.color = random.choice([
                (255, 215, 30), 
                (255, 223, 0),
                (255, 220, 100),
                (255, 251 ,2),
                (230, 255, 2),
                (72,255,0),
                (255,0,0),
                (255,0,242),
                (111,255,0),
                (245, 203, 241)
            ])
        else:
            self.color = random.choice([
                (0, 255, 238), 
                (40, 125, 252),
                (79, 64, 194),
                (0, 255 ,255),
                (10, 222, 190),
                (157, 195, 229)
            ])
        pygame.draw.circle(self.image, self.color, 
                         (self.size, self.size), self.size)  # 绘制圆形粒子
        
        self.rect = self.image.get_rect(center=position)
        
        # 运动向量分解
        self.velocity = [
            self.speed * math.cos(angle),
            self.speed * math.sin(angle)
        ]
        
        # 动态参数
        self.alpha = 255
        self.current_life = 0

    def update(self):
        """粒子运动逻辑"""
        # 运动衰减
        self.velocity[0] *= 0.92
        self.velocity[1] *= 0.92
        
        # 位置更新
        self.rect.x += self.velocity[0]
        self.rect.y += self.velocity[1]
        
        # 生命周期控制
        self.current_life += 1
        self.alpha = int(255 * (1 - self.current_life/self.lifetime))
        
        if self.current_life >= self.lifetime:
            self.kill()
        else:
            # 双重淡出效果（颜色+透明度）
            fade_image = self.image.copy()
            fade_image.fill((255,255,255, self.alpha), None, pygame.BLEND_RGBA_MULT)
            self.image = fade_image

class RhythmGame:
    """节奏游戏主控制器"""
    def __init__(self, audio_path):
        pygame.init()  # 初始化Pygame
        self.screen = pygame.display.set_mode((1280, 720))  # 创建1280x720窗口
        pygame.display.set_caption("This is true music")  # 设置窗口标题
        icon = pygame.image.load(resource_path("true.ico"))
        pygame.display.set_icon(icon)  # 设置窗口图标
        self.clock = pygame.time.Clock()  # 游戏时钟（控制帧率）
        self.load_assets()  # 加载游戏资源
        self.show_loading_screen()  # 显示加载界面
        self.setup_game_parameters(audio_path)  # 初始化游戏参数
        self.hit_effects = pygame.sprite.Group()
    def load_assets(self):
            """加载游戏资源"""
            self.notes = pygame.sprite.Group()  # 创建音符精灵组
            self.font = pygame.font.Font(None, 46)  # 创建字体对象（默认字体，36号）
            self.bgm=pygame.mixer.Sound(resource_path("bili_music.mp3"))  # 加载背景音乐
            # 加载并处理背景图片
            self.background = pygame.image.load(resource_path("bili_cover.jpg")).convert()
            self.background = pygame.transform.scale(self.background, (1280, 720))
            self.background.set_alpha(100)  # 设置半透明度（100/255）
            
    def show_loading_screen(self):
        """显示加载界面"""
        self.screen.blit(self.background, (0, 0))  # 绘制背景
        self.draw_centered_text('等待音频分析...别乱动...a s j k为按键...若未响应，等结束后按空格开始')  # 居中显示加载文本
        pygame.display.flip()  # 更新显示            
        
    def setup_game_parameters(self, audio_path):
        """配置核心游戏参数"""
        self.analyzer = AudioAnalyzer(audio_path)  # 创建音频分析器
        # 获取音符生成时间点和音乐速度
        self.note_times, self.tempo = self.analyzer.extract_features()
        
        # 动态速度调整（基于BPM）
        self.base_speed = 8  # 基准速度（120BPM对应的速度）
        # 修改原线性增速为指数缓增
        self.speed = self.base_speed * (1 + math.log1p(self.tempo / 100))
        
        # 游戏状态参数
        self.judge_line = 600  # 判定线Y坐标
        self.score = 0  # 当前得分
        self.combo = 0  # 当前连击数
        self.full=0
        self.max_combo = 0  # 最大连击数
        
        self.perfect = 150  # 完美判定阈值（±50ms）
        self.good = 300    # 良好判定阈值（±100ms）
        self.miss =100
        
        self.fix_times=0
        self.fix_time=0
        
        # 输入映射配置
        self.keys = [pygame.K_a, pygame.K_s, pygame.K_j, pygame.K_k]  # 可用按键
        self.lane_map = {  # 按键到轨道的映射
            pygame.K_a: 0,
            pygame.K_s: 1,
            pygame.K_j: 2,
            pygame.K_k: 3
        }
        
    def run(self):
            """启动游戏主流程"""
            self.wait_for_game_start()  # 等待玩家准备
            self.start_game_loop()  # 进入游戏循环
            
    def wait_for_game_start(self):
        """等待玩家按下开始键"""
        self.screen.fill((0, 0, 0))  # 清屏为黑色
        self.screen.blit(self.background, (0, 0))  # 绘制半透明背景
        self.draw_centered_text('按空格开始...')  # 显示提示文字
        pygame.display.flip()  # 更新显示
        
        start = False
        while not start:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # 处理退出事件
                    self.quit_game()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    start = True  # 检测空格键按下
                    
    def start_game_loop(self):
        """游戏主循环"""
        active_notes = deque(self.generate_notes())  # 创建音符队列
        self.bgm.play()  # 开始播放音乐
        self.start_time = pygame.time.get_ticks()  # 记录游戏开始时间
        
        running = True
        while running:
            current_time = pygame.time.get_ticks() - self.start_time  # 计算经过时间
            self.spawn_notes(active_notes, current_time)  # 生成新音符
            self.handle_events()  # 处理输入事件
            self.note_down()
            self.update_game_state(current_time)  # 更新游戏状态
            self.clock.tick(60)  # 保持60FPS
            
            
    def generate_notes(self):
        """生成音符数据队列"""
        note_types = ["tap"] * 7 + ["hold"]  # 音符类型概率分布（5:1）
        lanes = [0, 1, 2, 3]  # 可用轨道
        # 创建音符数据列表（时间单位转换为毫秒）
        return sorted([
            {
                "time": (t * 1000)-(620/self.speed)*(1000/60)+2000,  # 转换为毫秒
                "lane": random.choice(lanes),  # 随机分配轨道
                "type": random.choice(note_types)  # 随机选择类型
            } for t in self.note_times
        ], key=lambda x: x["time"])  # 按时间排序

    def spawn_notes(self, active_notes, current_time):
            """根据时间生成新音符"""
            # 生成未来2秒内需要出现的音符
            lanes = [0, 1, 2, 3]
            while active_notes and active_notes[0]["time"] <= current_time + 2000:
                note_data = active_notes.popleft()  # 从队列取出音符数据
                spawn_time = pygame.time.get_ticks()  # 记录实际生成时间
                # 创建Note对象并加入精灵组
                new_note = Note(
                    note_data["lane"], 
                    note_data["type"], 
                    self.speed,
                    spawn_time
                )
                self.notes.add(new_note)
                if note_data["type"]=="hold":
                    lanes.remove(note_data["lane"])
                    note_data["lane"]=random.choice(lanes)
                    new_note = Note(
                        note_data["lane"], 
                        note_data["type"], 
                        self.speed,
                        spawn_time
                    )
                    self.notes.add(new_note)
                    self.full+=300
                self.full+=300
    def handle_events(self):
        """处理系统事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # 窗口关闭事件
                self.quit_game()
            elif event.type == pygame.KEYDOWN and event.key in self.lane_map:
                self.process_input(event.key)  # 处理轨道按键
                
    def process_input(self, key):
            """处理玩家输入"""
            current_time = pygame.time.get_ticks()  # 获取当前精确时间（绝对时间）
            lane = self.lane_map[key]  # 获取对应轨道
            
            
            closest_note = None
            min_diff = float('inf')
            
            # 遍历所有音符检测命中
            for note in self.notes:
                # 新增轨道匹配检查
                if note.lane != lane:
                    continue
                
                # 修正时间差计算（使用绝对时间）
                if self.fix_times==11:
                    time_diff = abs(current_time - note.expected_time + self.fix_time)
                else:
                    time_diff = abs(current_time - note.expected_time)
                # 新增有效性检查（音符未过期）
                if time_diff > (self.good+self.miss):  # 音符过早出现
                    continue
                    
                # 寻找时间差最小的有效音符
                if abs(time_diff) <= self.good+self.miss and abs(time_diff) < min_diff:
                    min_diff = abs(time_diff)
                    closest_note = note
            
            if closest_note and min_diff <= self.good:
                self.evaluate_hit(min_diff, self.perfect, self.good, closest_note)
                min_diff = float('inf')
            elif min_diff <= self.good+self.miss:
                closest_note.kill()
                self.combo = 0  # 重置连击
                min_diff = float('inf')
                
                
    def evaluate_hit(self, time_diff, perfect, good, note):
        """评估命中精度"""
        if time_diff < perfect:
            self.handle_hit(300, note)  # 完美命中
        elif time_diff < good:
            self.handle_hit(100, note)  # 良好命中
            
            
    def handle_hit(self, score, note):
        """处理成功命中"""
        self.score += score  # 增加分数
        self.combo += 1  # 增加连击
        self.max_combo = max(self.max_combo, self.combo)  # 更新最大连击
        note.kill()  # 移除音符    
        # 生成粒子簇（完美命中更多粒子）
        particle_count = 50 if score == 300 else 20
        for _ in range(particle_count):
            particle = Particle(note.rect.center,level=particle_count)
            self.hit_effects.add(particle) 

    def note_down(self):
        if self.fix_times <10:
            for note in self.notes:
                if note.rect.top > 585 and note.fix==0:
                    note.fix=1
                    exp=note.expected_time
                    fact=pygame.time.get_ticks()
                    self.fix_time+=(exp-fact)
                    self.fix_times+=1
        elif self.fix_times==10:
            self.fix_time=self.fix_time/self.fix_times
            self.fix_times+=1
        for note in self.notes:
            if note.rect.top > 720:
                note.kill()
                self.combo = 0
    def update_game_state(self, current_time):
        """更新游戏每一帧的状态"""
        self.notes.update()  # 更新所有音符位置
        self.hit_effects.update()
        self.hit_effects.draw(self.screen)
        self.screen.blit(self.background, (0, 0))  # 重绘背景
        self.draw_judge_line()  # 绘制判定线
        self.notes.draw(self.screen)  # 绘制所有音符
        self.draw_hud(current_time)  # 绘制HUD信息
        pygame.display.flip()  # 刷新显示


    def draw_judge_line(self):
        """绘制判定线"""
        pygame.draw.line(self.screen, (255,215,000),  # 白色线条
                        (0, self.judge_line),  # 起点
                        (1280, self.judge_line), 7)  # 终点和线宽

    def draw_hud(self, current_time):
        """绘制游戏信息界面"""
        # 分数显示
        score_text = self.font.render(
            f"Score: {self.score} | Combo: {self.combo} (Max: {self.max_combo})   FULL:({self.full})", 
            True, (255,0,0))  # 红色文字
        self.screen.blit(score_text, (10, 10))  # 左上角位置
        
        # 进度条绘制
        progress = current_time / (self.analyzer.duration * 1000)  # 计算播放进度
        pygame.draw.rect(self.screen, (100,100,200),  # 蓝色进度条
                        (0, 670, 1280*progress, 30))  # 位置和尺寸
        if progress > 1:  # 播放结束
            self.quit_game()
            
            
    def draw_centered_text(self, text):
        """在屏幕中心绘制文本"""
        font = pygame.font.SysFont('dengxian', 40)  # 使用等线字体
        text_surf = font.render(text, True, (255, 255, 255))  # 白色文字
        text_rect = text_surf.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() // 2))
        self.screen.blit(text_surf, text_rect)


    def quit_game(self):
        """安全退出游戏"""
        pygame.quit()  # 关闭Pygame
        sys.exit()  # 退出程序

if __name__ == "__main__":
    # 游戏启动入口
    game = RhythmGame("./bili_music.mp3")  # 创建游戏实例
    game.run()  # 启动游戏