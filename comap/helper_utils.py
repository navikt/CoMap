from pandas import read_excel

def get_reduced_categories(file,sheet_name, exclude=[]):
    """
    Makes list and dictionary of categories in excelfile

    needs: filename, name of category sheet and master_cols
    
    optional: list of indices to exclude

    returns: list of columnnames and dictionary of columnnames
    """
    df_cat_sheet = read_excel(file,sheet_name)
    df_cat_sheet.reset_index(drop=True,inplace=True)

    cat_dict = {}
    categories = []
    for i,mc in enumerate(df_cat_sheet[sheet_name]):
        if i not in exclude:   
            categories.append(mc)
            cat_dict[i] = mc
    return cat_dict,categories