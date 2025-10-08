# test_dqn.py
import argparse
import os
import random
from typing import Tuple

import numpy as np
import torch
from torch import no_grad

from network.mlp import MLP as DQN


def select_action_greedy(q_net: torch.nn.Module, state: np.ndarray) -> int:
    """Chọn hành động greedy theo Q(s, a)."""
    with no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = q_net(state_t)
        action = int(torch.argmax(q_values).item())
    return action


def run_episode(env, q_net: torch.nn.Module, render: bool = False, max_steps: int = 10000) -> float:
    """Chạy 1 episode đánh giá, trả về tổng reward."""
    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        if render:
            env.render()

        action = select_action_greedy(q_net, state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward


def load_model(env, model_path: str, device: str = "cpu") -> torch.nn.Module:
    """Khởi tạo DQN cùng kích thước đúng và load trọng số từ file .pth."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = DQN(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=device)
    q_net.load_state_dict(checkpoint)
    q_net.eval()
    return q_net


def set_seed(seed: int):
    """Cố định seed cho reproducibility (ở mức cơ bản)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Test DQN đã train")
    parser.add_argument("--env-id", type=str, default="CartPole-v1", help="Gymnasium env id")
    parser.add_argument("--model-path", type=str, default="logs/dqn_cartpole.pth", help="Đường dẫn file .pth đã lưu")
    parser.add_argument("--episodes", type=int, default=10, help="Số episode test")
    parser.add_argument("--render", action="store_true", help="Bật render môi trường")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Import gymnasium ở đây để tránh phụ thuộc khi người dùng chỉ load file
    import gymnasium as gym

    # Tạo env
    env = gym.make(args.env_id)

    # Kiểm tra file model
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model tại '{args.model_path}'. "
            f"Hãy chắc chắn bạn đã train và lưu bằng train_dqn()."
        )

    # Load model
    q_net = load_model(env, args.model_path, device="cpu")

    # Chạy đánh giá
    rewards = []
    for ep in range(args.episodes):
        ep_ret = run_episode(env, q_net, render=args.render)
        rewards.append(ep_ret)
        print(f"[Eval] Episode {ep+1}/{args.episodes} - reward = {ep_ret:.2f}")

    avg_ret = float(np.mean(rewards)) if rewards else 0.0
    std_ret = float(np.std(rewards)) if rewards else 0.0
    print("-" * 60)
    print(f"✅ Average reward over {args.episodes} episodes: {avg_ret:.2f} ± {std_ret:.2f}")

    env.close()


if __name__ == "__main__":
    main()
