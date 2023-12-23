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

        self.task = 'us_city'
        self.setup_task(self.task)

    def setup_task(self, task):
        if task == 'us_city':
            import configs.scaling.US_city as scaling
            self.scaling = scaling
        elif task == 'ut_campus':
            import configs.scaling.UT_campus as scaling
            self.scaling = scaling

class LocalConfig(TACCConfig):
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

        self.task = 'us_city'
        self.setup_task(self.task)

TRAIN_CONFIGS = {
    'tacc': TACCConfig(),
    'local': LocalConfig()
}
