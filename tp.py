# import pandas as pd

# df = pd.read_csv(
#     "/Users/anujbohra/Desktop/Healthcare/TwoTower/Data/icd_codes_full_for_model.csv"
# )

# # Sanity check
# print(df[df["title"].str.contains("diabetes", case=False, na=False)].shape)

# # Clean text fields
# for col in ["synonyms", "chapter", "description", "title"]:
#     df[col] = df[col].fillna("")

# # Optional: save cleaned version
# df.to_csv(
#     "/Users/anujbohra/Desktop/Healthcare/TwoTower/Data/icd_codes_full_for_model_clean.csv",
#     index=False
# )


import pandas as pd

import pandas as pd
icd = pd.read_csv("/Users/anujbohra/Desktop/Healthcare/TwoTower/Data/icd_codes_full_for_model.csv")
lab = pd.read_csv("/Users/anujbohra/Desktop/Healthcare/TwoTower/Data/labels_scaled_nodot.csv")

print("overlap:", len(set(icd["code"].astype(str)) & set(lab["code"].astype(str))))


