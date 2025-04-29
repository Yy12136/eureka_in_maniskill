export OPENAI_API_BASE="https://ark.cn-beijing.volces.com/api/v3"
export OPENAI_API_KEY="f03c5260-8425-465c-b6c8-c929568a7e60"
export PYTHONPATH="${PYTHONPATH}:/home/yy/text2reward"

python metaworld_exp.py --TASK=drawer-open-v2
python metaworld_exp.py --TASK=drawer-close-v2
python metaworld_exp.py --TASK=window-open-v2
python metaworld_exp.py --TASK=window-close-v2
python metaworld_exp.py --TASK=button-press-v2
python metaworld_exp.py --TASK=sweep-into-v2
python metaworld_exp.py --TASK=door-unlock-v2
python metaworld_exp.py --TASK=door-close-v2
python metaworld_exp.py --TASK=handle-pull-v2
python metaworld_exp.py --TASK=handle-press-v2
python metaworld_exp.py --TASK=handle-press-side-v2