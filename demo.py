from utils import get_config

setup config
cfg = get_config()
cfg.merge_from_file('configs/text_recog.yaml')

moran = cfg.TEXT_RECOG.MORAN