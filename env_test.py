import jax
import jax.numpy as jnp
from jax import random
import sys

print("--- JAX GPU 验证脚本 (修正版) ---")
print(f"JAX 版本: {jax.__version__}")
print(f"Python 版本: {sys.version.split()[0]}")

print("\n--- 1. 检查 JAX 后端 ---")
try:
    default_backend = jax.default_backend()
    print(f"JAX 默认后端: {default_backend}")
except Exception as e:
    print(f"获取默认后端时出错: {e}")
    default_backend = "cpu" # 假设失败

if default_backend != 'gpu':
    print("\n!! 警告: JAX 默认后端不是 'gpu'。")
    print("这可能意味着 GPU 插件未正确加载或 CUDA 不兼容。")
    print("将尝试手动查找设备...\n")

print("\n--- 2. 列出所有 JAX 设备 ---")
try:
    devices = jax.devices()
    if not devices:
        print("!! 错误: jax.devices() 返回了一个空列表。")
    else:
        print(f"共找到 {len(devices)} 个设备:")
        for i, dev in enumerate(devices):
            print(f"  设备 {i}: {dev.platform} (设备 ID: {dev.id}) - {dev.device_kind}")

    print("\n--- 3. 筛选 GPU 设备 ---")
    # 查找 'gpu' 平台的设备
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    
    if gpu_devices:
        print(f"\n[成功]：成功检测到 {len(gpu_devices)} 个 GPU！")
        
        print("\n--- 4. 在第一个 GPU 上执行计算测试 ---")
        try:
            # 将数据显式移动到第一个 GPU
            gpu_device = gpu_devices[0]
            print(f"正在使用设备: {gpu_device.device_kind}")
            
            key = jax.random.PRNGKey(42)
            size = 3000
            
            # 在 CPU 上创建数据
            x_cpu = jax.random.normal(key, (size, size))
            # 【修正】: .device() 改为 .device
            print(f"数据在 CPU 上的设备: {x_cpu.device}") 
            
            # 将数据显式移动到 GPU
            print("正在将数据移动到 GPU...")
            x_gpu = jax.device_put(x_cpu, device=gpu_device)
            # 【修正】: .device() 改为 .device
            print(f"数据在 GPU 上的设备: {x_gpu.device}")

            # 使用 JIT 编译并运行矩阵乘法
            print("正在 JIT 编译并执行矩阵乘法 (a @ a)...")
            result = jax.jit(lambda a: a @ a, device=gpu_device)(x_gpu)
            
            # block_until_ready() 确保计算完成
            result.block_until_ready()
            
            print("计算完成。")
            # 【修正】: .device() 改为 .device
            print(f"结果数据所在设备: {result.device}")

            # 【修正】: .device() 改为 .device
            if result.device.platform == 'gpu':
                print("\n*** 恭喜！JAX GPU 配置成功！***")
            else:
                print("\n!! 警告: 计算结果不在 GPU 上，请检查配置。")

        except Exception as e:
            print(f"\n!! GPU 计算测试失败: {e}")
            
    else:
        print("\n[失败]：未检测到 GPU 设备。")
        print("JAX 将仅在 CPU 上运行。")
        print("请确保：")
        print("  1. 您在 WSL 2 (Ubuntu) 终端中运行。")
        print("  2. 您 Windows 主机的 NVIDIA 驱动已正确安装。")
        print("  3. 您已正确安装了 jax-cuda12-plugin。")

except Exception as e:
    print(f"\n!! 验证脚本发生严重错误: {e}")
    print("JAX GPU 配置可能不正确。")



key = random.PRNGKey(0)
x = random.normal(key, (5000, 5000))
y = random.normal(key, (5000, 5000))

# ----------------------------------------------------
# 方式一：糟糕的方式 (在 CPU 上运行, 非常慢)
# ----------------------------------------------------
def slow_cpu_multiply(a, b):
    # 即使 JAX 看到了 GPU，这个非 JIT 的函数
    # 也很可能在 CPU 上逐行“模拟”执行
    return jnp.dot(a, b)

# result_cpu = slow_cpu_multiply(x, y).block_until_ready()


# ----------------------------------------------------
# 方式二：正确的方式 (在 GPU 上编译并运行, 非常快)
# ----------------------------------------------------
@jax.jit # 加上这个装饰器，一切都不同了
def fast_gpu_multiply(a, b):
    # XLA 会把这个函数编译成一个 CUDA 内核
    return jnp.dot(a, b)

# JAX 会自动将 x 和 y 复制到 GPU 显存,
# 在 GPU 上执行编译好的内核, 然后将结果返回
result_gpu = fast_gpu_multiply(x, y).block_until_ready()

print(f"JAX 默认后端: {jax.default_backend()}")
print(f"结果所在设备: {result_gpu.device}")

print("\n--- 验证结束 ---")