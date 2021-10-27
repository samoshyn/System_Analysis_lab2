import streamlit as st
import numpy as np
import pandas as pd
from polynomial_builder import PolynomialBuilder
from solve import Solve
import time

st.set_page_config(layout="wide")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)



def search_params(df):
    
    st_x1s = [i for i in range(2, 10, 2)]
    st_x2s = [i for i in range(2, 10, 2)]
    st_x3s = [i for i in range(2, 10, 2)]
    opt_err = np.inf
    opt_dct = dict()
    all_results = []
    with st.spinner('Ми працюємо, зачекайте...'):
        for st_x1 in st_x1s:
            for st_x2 in st_x2s:
                for st_x3 in st_x3s:
                    #if st_x1 + st_x2 + st_x3 > 3:
                    dct = {
                    'poly_type': st.session_state['method'], # 
                    'input_file': st.session_state['df'],
                    'samples': st.session_state['samples'],
                    'dimensions': [st.session_state[i] for i in ['dim_x1', 'dim_x2', 'dim_x3', 'dim_y']], # rozmirnist' vectoriv (x1,x2,x3,y)
                    'degrees': [st_x1, st_x2, st_x3], # stepin' polinoma
                    'weights': st.session_state['weights'], # vagi (scaled/average)
                    'lambda_multiblock': int(st.session_state['lambda_multiblock']),
                        }
                    solver = Solve(dct)
                    buffer, err = solver.prepare()
                    solution = PolynomialBuilder(solver)
                    y = np.array(solution._solution.Y_)
                    f = np.array(solution._solution.F_)
                    y_norm = y / np.max(y)
                    f_norm = f / np.max(f)
                    err = np.mean(np.mean(abs(y_norm - f_norm), axis=0))
                    all_results.append({"degrees": ', '.join([str(st_x1), str(st_x2), str(st_x3)]),
                                        "error": err})
                    print(err)
                    if err < opt_err:
                        opt_err = err
                        opt_dct = dct
                        
    st.success('Все готово! 10 найкращих ітерацій виведено нижче')
    st.dataframe(pd.DataFrame(all_results).sort_values(by='error').iloc[:10])
    print('Top', opt_err)
    for i, step in zip(['st_x1', 'st_x2', 'st_x3'], opt_dct['degrees']):
        st.session_state[i] = step
                            
                    
    
def config_params(df):
    
    samples = st.slider('Розмір вибірки', min_value=1, max_value=len(df), value=len(df), key='samples')
    dim_x1 = st.slider('Розмірність Х1', min_value=1, max_value=5, key='dim_x1')
    dim_x2 = st.slider('Розмірність Х2', min_value=1, max_value=5, key='dim_x2')
    dim_x3 = st.slider('Розмірність Х3', min_value=1, max_value=5, key='dim_x3')
    dim_y = st.slider('Розмірність Y', min_value=1, max_value=5, key='dim_y')
    if sum([st.session_state[i] for i in ['dim_x1', 'dim_x2', 'dim_x3', 'dim_y']])>df.shape[1]:
        st.session_state['norm_params'] = False
        st.error('Перевищена сумарна розмірність вибірки. Будь-ласка, змініть параметри для подальшої роботи')
    else:
        st.session_state['norm_params'] = True
        method = st.radio('Вид поліномів', ["Поліноми Чебишева", "Поліноми Лежандра", 
                                            "Поліноми Лагерра", "Поліноми Ерміта"], key='method')
        grid_search = st.radio('Ввести степені вручную чи підібрати найкращі автоматично?', ["Вручну", "Підібрати"], key='grid_search')
        if grid_search=='Вручну':
            if method:
                st_x1 = st.slider('Степінь полінома для Х1', min_value=1, max_value=9, key='st_x1')
                st_x2 = st.slider('Степінь полінома для Х2', min_value=1, max_value=9, key='st_x2')
                st_x3 = st.slider('Степінь полінома для Х3', min_value=1, max_value=9, key='st_x3')
                
            weights = st.radio('Вид вагів', ["Нормоване", "Середнє"], key='weights')
            lambda_multiblock = st.checkbox('Визначити лямбда через 3 системи', key='lambda_multiblock')
        else:
            st.write('Оберіть додаткові параметри та натисність кнопку "Оптимізувати степені"')
            weights = st.radio('Вид вагів', ["Нормоване", "Середнє"], key='weights')
            lambda_multiblock = st.checkbox('Визначити лямбда через 3 системи', key='lambda_multiblock')
            opt = st.button('Оптимізувати степені')
            if opt:
                search_params(df)
        
        
def get_df(file):
    
  extension = file.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(file, header=None, sep='\t')
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(file, header=None, engine='openpyxl')

  return df

def plots(solution):
    
    st.header('Графіки')
    cols = st.columns(solution._solution.Y.shape[1])
    for i, col in enumerate(cols):
        time.sleep(0.1)
        with col:
            col.subheader(f'Координата Y{i+1}')
            y = np.array(solution._solution.Y_[:, i]).reshape(-1,)
            f = np.array(solution._solution.F_[:, i]).reshape(-1,)
            y_norm = y / max(y)
            f_norm = f / max(f)
            err = abs(y_norm - f_norm)
            col.line_chart({"Y": y,
                           "F": f})
            col.subheader(f'Координата Y{i+1}_norm')
            col.line_chart({"Y_norm": y_norm,
                           "F_norm": f_norm})
            col.subheader(f'Помилка {i+1}')
            col.line_chart({"err": err})
            col.write(f'**Середня помилка**: {np.mean(err)}')

def main():
    st.sidebar.title("Виконуйте покроково")
    select_menu = st.sidebar.radio("Етапи ", ["Налаштовуємо алгоритм", "Досліджуємо результати"])
    df = None
    buffer = None
    print_result = None
    if select_menu == "Налаштовуємо алгоритм":
        
        st.header('Крок 1. Дані')
        st.info("Завантажте датасет для подальшого налаштування алгоритму")
        uploaded_file = st.file_uploader("Завантажити вхідні дані")
        if uploaded_file is not None:
            df = get_df(uploaded_file)
            st.session_state['df'] = df            
            
        if st.session_state.get('df') is not None or st.session_state.get('print_result') is not None:
            st.header('Крок 2. Налаштування')
            config_params(st.session_state['df'])
            if st.session_state.get('norm_params'):
                confirm_params = st.button('Запустити алгоритм')
                if confirm_params:
                    
                    dct = {
                            'poly_type': st.session_state['method'], # 
                            'input_file': st.session_state['df'],
                            'samples': st.session_state['samples'],
                            'dimensions': [st.session_state[i] for i in ['dim_x1', 'dim_x2', 'dim_x3', 'dim_y']], # rozmirnist' vectoriv (x1,x2,x3,y)
                            'degrees': [st.session_state[i] for i in ['st_x1', 'st_x2', 'st_x3']], # stepin' polinoma
                            'weights': st.session_state['weights'], # vagi (scaled/average)
                            'lambda_multiblock': int(st.session_state['lambda_multiblock']),
                        }
                    solver = Solve(dct)
                    buffer, err = solver.prepare()
                    st.session_state['buffer'] = buffer
                    
                    solution = PolynomialBuilder(solver)
                    print_result = solution.get_results()
                    st.session_state['print_result'] = print_result
                    st.session_state['solution'] = solution
                    if print_result:
                        st.success('Успіх! Переходьте у розділ "Досліджуємо результати"')
                        st.balloons()


    if select_menu == "Досліджуємо результати":
        
        if st.session_state.get('print_result') is None:
            st.error("Ви не запустили алгоритм. Поверніться на попередній крок")
        else:
            st.header('Результати роботи алгоритму')
            title = st.text_input('Введіть назви 2 файлів через пробіл: для результуючих матриць та многочленів. Після цього натисніть Enter', 'results polinoms')
            st.download_button(
                    label="Натисніть для вивантаження результуючих матриць",
                    data=st.session_state['buffer'],
                    file_name=f"{title.split()[0]}.xlsx",
                    mime="application/vnd.ms-excel"
                    )
            
            plots(st.session_state['solution'])
            
            st.download_button(
                    label="Натисніть для вивантаження многочленів",
                    data=st.session_state['print_result'].replace('**','').replace('\\',''),
                    file_name=f"{title.split()[1]}.txt")
            
            st.write(st.session_state['print_result'])
            
main()
