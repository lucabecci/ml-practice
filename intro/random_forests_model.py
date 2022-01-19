import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def model_process():
    m_file_path = './input/melb_data.csv'
    data = pd.read_csv(m_file_path)
    filtered_data = data.dropna(axis=0)
    y = filtered_data.Price
    featured_list = [
        'Rooms', 'Bathroom', 'Landsize', 
        'BuildingArea','YearBuilt', 'Lattitude', 'Longtitude'
    ]
    x = filtered_data[featured_list]
    t_x, v_x, t_y, v_y = train_test_split(x, y, random_state=0)
    model = RandomForestClassifier(random_state=1)
    model.fit(t_x, t_y)
    predict_val = model.predict(v_x)
    mae = mean_absolute_error(v_y, predict_val)
    print("MAE WITH RANDOM FORES:", mae)


model_process()