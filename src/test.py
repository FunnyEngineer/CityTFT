res_df = normalize_load(res_df, self.h_mean, self.h_std, self.c_mean, self.c_std)
        bud_df = read_building_info(self.buildin