import matplotlib.pyplot as plt
import numpy as np

def plot_rewards_split(file_path):
    """
    创建两个子图：左侧是三个智能体的奖励曲线，右侧是总奖励曲线
    """
    epochs = []
    reward_0 = []
    reward_1 = []
    reward_2 = []
    reward_total = []
    
    # 读取数据
    with open(file_path, 'r') as f:
        # 跳过第一行（标题行）
        next(f)
        
        for line in f:
            data = line.strip().split()
            if len(data) >= 5:
                epochs.append(int(data[0]))
                reward_0.append(float(data[1]))
                reward_1.append(float(data[2]))
                reward_2.append(float(data[3]))
                reward_total.append(float(data[4]))
    
    # 创建左右分裂的两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle('Multi-Agent SAC Training Progress', fontsize=18, fontweight='bold')
    
    # 左侧子图 - 三个智能体的奖励曲线
    ax1.plot(epochs, reward_0, 'b-', linewidth=2, label='Agent 0')
    ax1.plot(epochs, reward_1, 'r-', linewidth=2, label='Agent 1')
    ax1.plot(epochs, reward_2, 'g-', linewidth=2, label='Agent 2')
    ax1.set_title('Individual Agent Rewards', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Reward Sum', fontsize=14)
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(epochs))
    
    # 计算个体智能体奖励的平均值
    avg_0 = np.mean(reward_0)
    avg_1 = np.mean(reward_1)
    avg_2 = np.mean(reward_2)
    
    # 在左图添加平均值水平线
    ax1.axhline(y=avg_0, color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_1, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(y=avg_2, color='green', linestyle='--', alpha=0.5)
    
    # 右侧子图 - 总奖励曲线
    ax2.plot(epochs, reward_total, 'm-', linewidth=3)
    ax2.set_title('Total Reward (All Agents)', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Total Reward Sum', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(epochs))
    
    # 添加总奖励统计信息
    avg_total = np.mean(reward_total)
    max_total = np.max(reward_total)
    ax2.axhline(y=avg_total, color='gray', linestyle='--', alpha=0.7, 
                label=f'Avg: {avg_total:.0f}')
    ax2.axhline(y=max_total, color='gray', linestyle='-', alpha=0.7, 
                label=f'Max: {max_total:.0f}')
    ax2.legend(fontsize=12)
    
    # 添加数据点标记以便更好地可视化
    for ax, data in [(ax1, [reward_0, reward_1, reward_2]), (ax2, [reward_total])]:
        for i, y in enumerate(data):
            color = ['blue', 'red', 'green', 'magenta'][i]
            ax.scatter(epochs, y, color=color, s=30, alpha=0.6)
    
    # 在图中添加一些统计文本
    stat_text = f"Training Statistics:\n" \
                f"Epochs: {len(epochs)}\n" \
                f"Avg Agent 0: {avg_0:.0f}\n" \
                f"Avg Agent 1: {avg_1:.0f}\n" \
                f"Avg Agent 2: {avg_2:.0f}\n" \
                f"Avg Total: {avg_total:.0f}\n" \
                f"Max Total: {max_total:.0f}"
    fig.text(0.98, 0.15, stat_text, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8), ha='right')
    
    # 保存并显示
    plt.tight_layout()
    plt.savefig('masac_rewards_split.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print(f"训练进度: {len(epochs)} epochs")
    print(f"Agent 0 - 平均: {avg_0:.2f}")
    print(f"Agent 1 - 平均: {avg_1:.2f}")
    print(f"Agent 2 - 平均: {avg_2:.2f}")
    print(f"总奖励 - 平均: {avg_total:.2f}, 最大: {max_total:.2f}")

if __name__ == "__main__":
    file_path = "./data/ma_epoch_rewards.txt"
    plot_rewards_split(file_path)