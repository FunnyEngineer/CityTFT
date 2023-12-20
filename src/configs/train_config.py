class TACCConfig():
    def __init__(self) -> None:
        self.accleator = 'gpu'
        self.precision = '16-mixed'
        self.devices = 1
        self.cli_dir = '/work/08388/tudai/ls6/US_cities/climate/historic'
        self.res_dir = '/work/08388/tudai/ls6/US_cities/result'
        self.bud_dir = '/work/08388/tudai/ls6/US_cities/bud'
        self.ref_csv = '/work/08388/tudai/ls6/US_cities/ref.csv'
        self.cli_scaler = '/work/08388/tudai/ls6/US_cities/climate_scaler.pkl'

        self.num_workers = 31

class LocalConfig():
    def __init__(self) -> None:
        self.accleator = 'gpu'
        self.precision = '16-mixed'
        self.devices = 1
        self.cli_dir = '../US_cities/climate/historic'
        self.res_dir = '../US_cities/result'
        self.bud_dir = '../US_cities/bud'
        self.ref_csv = '../US_cities/ref.csv'
        self.cli_scaler = '../US_cities/climate_scaler.pkl'

        self.num_workers = 20

TRAIN_CONFIGS = {
    'tacc': TACCConfig(),
    'local': LocalConfig()
}
