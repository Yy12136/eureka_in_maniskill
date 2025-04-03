export OPENAI_API_BASE="https://ark.cn-beijing.volces.com/api/v3"
export OPENAI_API_KEY="f03c5260-8425-465c-b6c8-c929568a7e60"
export PYTHONPATH="${PYTHONPATH}:/home/yy/text2reward"

python zero_shot_exp.py --TASK=LiftCube-v0
python zero_shot_exp.py --TASK=PickCube-v0
python zero_shot_exp.py --TASK=TurnFaucet-v0
python zero_shot_exp.py --TASK=OpenCabinetDoor-v1
python zero_shot_exp.py --TASK=OpenCabinetDrawer-v1
python zero_shot_exp.py --TASK=PushChair-v1