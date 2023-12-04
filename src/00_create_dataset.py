from utils.misc import *
from pathlib import Path
import pdb
from bs4 import BeautifulSoup
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union, polygonize
import matplotlib.pyplot as plt

def main():
    cli_dir = './new_cli/citydnn'
    building_path = './data/ut_building_info.csv'
    res_dir = './data/citydnn'
    export_dir = './data/citydnn/nn'


    heat_key = 'Heating(Wh)'
    cool_key = 'Cooling(Wh)'

    cli_dir = Path(cli_dir)
    cli_list = list(cli_dir.glob('**/*.cli'))
    res_dir = Path(res_dir)

    bud_df = read_building_info(building_path)

    
    for cli_file in cli_list:
        cli_df = read_climate_file(cli_file)
        res_file = res_dir / cli_file.stem / f'{cli_file.stem}_TH.out'
        res_df = read_result_file(res_file)

        total_df = pd.DataFrame()
        # get building id
        for bud_id in bud_df.id:
            for t in range(len(cli_df)):
                row = {} # init row

                row = bud_df.loc[bud_id].to_dict() # add building property to row
                cli_prop = cli_df.loc[t].to_dict()
                row.update(cli_prop)
                row[heat_key] = res_df.loc[0, res_df.columns.str.startswith(f'{bud_id}(1)') & res_df.columns.str.endswith(heat_key)].item()
                row[cool_key] = res_df.loc[0, res_df.columns.str.startswith(f'{bud_id}(1)') & res_df.columns.str.endswith(cool_key)].item()
            
                total_df = pd.concat([total_df, pd.DataFrame(row, index=[len(total_df)])], ignore_index=True)
                pdb.set_trace()
        total_df.to_csv(f'{export_dir}/{cli_file.stem}.csv', index=False)
        
def from_xy_to_aspect_ratio(x, y):
    edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
    return max(edge_length) / min(edge_length)

def xml_to_df(xml_path, export_path):

    def attrs_list_to_numeric_filter_series(attrs_list):
        df = pd.DataFrame(attrs_list)
        df = df.apply(pd.to_numeric)
        return dict(df.mean().dropna().drop(labels=['id', 'type']))

    with open(xml_path, 'r') as f:
        html = f.read()
        
    soup = BeautifulSoup(html, 'xml')

    building_list = soup.find_all('Building')

    bu_attr_list = []
    for bu in building_list:
        bu_attrs = bu.attrs # init building attrs
        
        # building level
        ht = bu.find('HeatTank')
        ht_attrs = ht.attrs
        for attr_name in ht_attrs:
            bu_attrs['HeatTank_' + attr_name] = ht_attrs[attr_name]
        ct = bu.find('CoolTank')
        ct_attrs = ct.attrs
        for attr_name in ct_attrs:
            bu_attrs['CoolTank_' + attr_name] = ct_attrs[attr_name]
        
        hs_hp = bu.find('HeatSource').find('HeatPump')
        hs_hp_attrs = hs_hp.attrs
        for attr_name in hs_hp_attrs:
            bu_attrs['HeatSource_HeatPump_' + attr_name] = hs_hp_attrs[attr_name]
        
        cs_hp = bu.find('CoolSource').find('HeatPump')
        cs_hp_attrs = cs_hp.attrs
        for attr_name in cs_hp_attrs:
            bu_attrs['CoolSource_HeatPump_' + attr_name] = cs_hp_attrs[attr_name]
        
        # Zone level
        zone = bu.find('Zone')
        zo_attrs = zone.attrs
        for attr_name in zo_attrs:
            bu_attrs['Zone_' + attr_name] = zo_attrs[attr_name]
            
        occu_attrs = zone.find('Occupants').attrs
        for attr_name in occu_attrs:
            bu_attrs['Occupants_' + attr_name] = occu_attrs[attr_name]
        
        # Wall level
        wall_list = zone.find_all('Wall')
        w_attrs_list = []
        perimeter = 0
        w_polygon_list = []
        all_cor_list = []
        for wall in wall_list:
            w_attrs = wall.attrs

            # add Uvalue
            try:
                type_id = w_attrs['type']
            except:
                pdb.set_trace()
            w_Uvalue = float(soup.find('Composite', id=type_id).find_all('Layer')[2].attrs['Conductivity']) * 10
            w_attrs['UValue'] = w_Uvalue

            # get wall perimeter
            v_node = [v for v in wall.contents if v.name] # get V0, V1, V2, V3
            cor_list = [(float(v.attrs['x']), float(v.attrs['y'])) for v in v_node]
            w_polygon = Polygon(cor_list) # extract x, y value
            all_cor_list += cor_list
            w_polygon_list.append(w_polygon)
            perimeter += w_polygon.length / 2 # add wall length to perimeter (/2 because of double counting)

            w_attrs_list.append(w_attrs)
        big_p = Polygon(all_cor_list)
        aspect_ratio = from_xy_to_aspect_ratio(*big_p.minimum_rotated_rectangle.exterior.coords.xy)
            
        w_attrs = attrs_list_to_numeric_filter_series(w_attrs_list)
        for attr_name in w_attrs:
            bu_attrs['Wall_' + attr_name] = w_attrs[attr_name]
        bu_attrs['Wall_perimeter'] = perimeter # Footprint perimeter
        bu_attrs['Wall_aspect_ratio'] = aspect_ratio # Aspect ratio
        bu_attrs['Footprint_area'] = big_p.convex_hull.area # Footprint area

        # Roof level
        roof_list = zone.find_all('Roof')
        r_attrs_list = []
        r_z = []
        for roof in roof_list:
            r_attrs = roof.attrs
            type_id = r_attrs['type']
            r_Uvalue = soup.find('Composite', id=type_id).attrs['Uvalue']
            r_z.append(float(roof.V1.attrs['z']))
            r_attrs['UValue'] = r_Uvalue
            r_attrs_list.append(r_attrs)
        r_attrs = attrs_list_to_numeric_filter_series(r_attrs_list)
        for attr_name in r_attrs:
            bu_attrs['Roof_' + attr_name] = r_attrs[attr_name]
        bu_attrs['Roof_z'] = max(r_z)

        floor_list = zone.find_all('Floor')
        f_attrs_list = []
        for floor in floor_list:
            f_attrs = floor.attrs

            type_id = f_attrs['type']
            # add Uvalue
            f_Uvalue = soup.find('Composite', id=type_id).attrs['Uvalue']
            f_attrs['UValue'] = f_Uvalue
            
            f_attrs_list.append(f_attrs)
        f_attrs = attrs_list_to_numeric_filter_series(f_attrs_list)
        for attr_name in f_attrs:
            bu_attrs['Floor_' + attr_name] = f_attrs[attr_name]
        
        bu_attr_list.append(bu_attrs)
    
    df = pd.DataFrame(bu_attr_list)
    df.to_csv(export_path, index=False)

if __name__ == '__main__':
    # main()
    # iterate random xml files
    xml_dir = Path('./data/random_urban')
    file_list = xml_dir.glob('*.xml')
    for f in file_list:
        export_path = f.parent / (f.stem + '.csv')
        xml_to_df(f, export_path)