# -*- coding: utf-8 -*-
"""Created on Mon Dec 16 16:45:04 2019@author: Alexm"""




def get_all_csv_files_from_zip(fp,suffix=".csv"):
    import pandas as pd
    from zipfile import ZipFile
    with ZipFile(fp, 'r') as zip_obj: 
         zip_df_dict = {}
         for text_file in zip_obj.infolist():
             if text_file.filename.endswith(suffix):
                zip_df_dict[text_file.filename] = pd.read_csv(zip_obj.open(text_file.filename))
    return zip_df_dict    



movie_dfdict = get_all_csv_files_from_zip(r"C:\Users\Alexm\Desktop\ml-20m.zip")

mv_list = movie_dfdict["ml-20m/ratings.csv"]

all_users_count = mv_list["userId" ].value_counts().sort_index()
all_films_count = mv_list["movieId"].value_counts().sort_index()

val = all_films_count.sort_values(ascending=False).iloc[1000]
all_films_count = all_films_count[all_films_count.apply(lambda x:x>=val)]



