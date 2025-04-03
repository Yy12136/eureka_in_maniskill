import mani_skill2
from mani_skill2.utils.registration import get_env_list

# 获取所有环境列表
env_list = get_env_list()
print("\nAvailable environments:")
for env_id in env_list:
    print(f"- {env_id}")