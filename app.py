from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Lista predefinida de todas las agrupaciones políticas conocidas
todas_agrupaciones = [
    "ACCION POPULAR", "ALIANZA PARA EL PROGRESO", "AVANZA PAIS - PARTIDO DE INTEGRACION SOCIAL",
    "DEMOCRACIA DIRECTA", "EL FRENTE AMPLIO POR JUSTICIA, VIDA Y LIBERTAD", "JUNTOS POR EL PERU",
    "PARTIDO DEMOCRATICO SOMOS PERU", "PODEMOS POR EL PROGRESO DEL PERU", "SOLIDARIDAD NACIONAL",
    "VAMOS PERU", "AVANZADA REGIONAL INDEPENDIENTE UNIDOS POR HUANUCO", "MOVIMIENTO POLITICO CAMBIEMOS X HCO",
    "MOVIMIENTO POLITICO FRENTE AMPLIO REGIONAL PAISANOCUNA", "MOVIMIENTO POLITICO HECHOS Y NO PALABRAS",
    "AUTENTICO REGIONAL", "MOVIMIENTO INTEGRACION DESCENTRALISTA", "MOVIMIENTO INDEPENDIENTE REGIONAL LUCHEMOS POR HUANUCO",
    "UNION POR EL PERU", "FUERZA POPULAR","FRENTE DEMOCRATICO REGIONAL"
]
app = Flask(__name__)
CORS(app)



# Cargar los datos al iniciar la aplicación
data_files = {
    '2014': pd.read_excel('2014/ERM2014_Actas.xlsx'),
    '2014-v2': pd.read_excel('2014/ERM2014-2V_Actas.xlsx'),
    '2018': pd.read_excel('2018/ERM2018_Actas.xlsx'),
    '2022': pd.read_excel('2022/ERM2022_Actas.xlsx'),
    '2022-g':pd.read_excel('2022/ERM2022_Padron_Regional_filtro.xlsx'),
    '2018-g':pd.read_excel('2018/ERM2018_Padron_Regional_filtro.xlsx'),
    '2014-g':pd.read_excel('2014/ERM2014_Padron_Regional_filtro.xlsx'),
    '2014-v2-g':pd.read_excel('2014/ERM2014-2V_Padron_Regional_filtro.xlsx'),
    '2014-r': pd.read_excel('2014/ERM2014_Resultados_Regional_filtro.xlsx'),
    '2014-v2-r': pd.read_excel('2014/ERM2014-2V_Resultados_Regional_filtro.xlsx'),
    '2018-r': pd.read_excel('2018/ERM2018_Resultados_Regional_filtro.xlsx'),
    '2022-r': pd.read_excel('2022/ERM2022_Resultados_Regional_filtro.xlsx'),
}



@app.route('/analisis', methods=['GET'])
def analisis():
    limit = int(request.args.get('limit', 10))  # Valor por defecto 10
    offset = int(request.args.get('offset', 0))  # Valor por defecto 0
    provincia = request.args.get('provincia', None)  # Valor por defecto None si no se proporciona
    datayear = request.args.get('datayear', '2022')

    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022']

    # Agrupar datos por mesa y sumar/agregar apropiadamente
    grouped_data = data.groupby(['UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'MESA']).agg({
        'VOTOS_OBTENIDOS': 'sum',
        'ELECTORES_HABILES': 'first',  # El número de electores habilitados debe ser el mismo para cada entrada de la misma mesa
        'VOTOS_BLANCOS': 'first',  # Tomar el primer registro
        'VOTOS_NULOS': 'first',  # Tomar el primer registro
        'VOTOS_IMPUG': 'first'  # Tomar el primer registro
    }).reset_index()

    # Calcular la suma total de votos por cada mesa
    grouped_data['TOTAL_VOTOS'] = (grouped_data['VOTOS_OBTENIDOS'] +
                                   grouped_data['VOTOS_BLANCOS'] +
                                   grouped_data['VOTOS_NULOS'] +
                                   grouped_data['VOTOS_IMPUG'])

    # Calcular el número de personas que no votaron
    grouped_data['NO_VOTANTES'] = grouped_data['ELECTORES_HABILES'] - grouped_data['TOTAL_VOTOS']

    # Añadir una columna para verificar si la suma de votos coincide con los electores habilitados
    grouped_data['VERIFICACION'] = (grouped_data['TOTAL_VOTOS'] + grouped_data['NO_VOTANTES']) == grouped_data['ELECTORES_HABILES']

    if provincia:
        grouped_data = grouped_data[grouped_data['PROVINCIA'].str.contains(provincia, case=False, na=False)]

    # Convertir el DataFrame a JSON
    limited_data = grouped_data.iloc[offset:offset + limit]
    result = limited_data.to_dict(orient="records")

    return jsonify(result)

@app.route('/analisis_provincia', methods=['GET'])
def analisis_provincia():
    departamento = request.args.get('departamento', None)
    datayear = request.args.get('datayear', '2022')

    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022']

    grouped_data_obtenidos = data.groupby(['DEPARTAMENTO', 'PROVINCIA']).agg({
        'VOTOS_OBTENIDOS': 'sum',
    }).reset_index()
    data = data.drop_duplicates(subset=['MESA'])

    # Agrupar los datos por departamento y provincia
    grouped_data = data.groupby(['DEPARTAMENTO', 'PROVINCIA']).agg({
        'ELECTORES_HABILES': 'sum',
        'VOTOS_BLANCOS': 'sum',
        'VOTOS_NULOS': 'sum',
        'VOTOS_IMPUG': 'sum'
    }).reset_index()

    # Calcular el total de no votantes por provincia
    grouped_data['TOTAL_VOTOS'] = (
        grouped_data['VOTOS_BLANCOS'] +
        grouped_data['VOTOS_NULOS'] +
        grouped_data_obtenidos['VOTOS_OBTENIDOS'] +
        grouped_data['VOTOS_IMPUG']
    )

    grouped_data['NO_VOTANTES'] = grouped_data['ELECTORES_HABILES'] - grouped_data['TOTAL_VOTOS']
    grouped_data['VOTOS_OBTENIDOS_TOTAL'] = grouped_data_obtenidos['VOTOS_OBTENIDOS']

    if departamento:
        grouped_data = grouped_data[grouped_data['DEPARTAMENTO'].str.contains(departamento, case=False, na=False)]

    result = grouped_data.to_dict(orient="records")

    return jsonify(result)

@app.route('/analisis_provincia-politicos', methods=['GET'])
def analisis_provincia_politicos():
    departamento = request.args.get('departamento', None)
    datayear = request.args.get('datayear', '2022')

    # Cargar los datos del año especificado o por defecto 2022
    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022']

    # Filtrar los datos por departamento si se especifica uno
    if departamento:
        data = data[data['DEPARTAMENTO'].str.contains(departamento, case=False, na=False)]

    # Agrupar datos por departamento, provincia, y agrupaciones políticas
    data_agrupada = data.groupby(['DEPARTAMENTO', 'PROVINCIA', 'AGRUPACION_POLITICA'])['VOTOS_OBTENIDOS'].sum().reset_index()

    # Reestructurar datos para incluir todas las agrupaciones conocidas
    resultado_final = defaultdict(lambda: defaultdict(dict))
    for index, row in data_agrupada.iterrows():
        resultado_final[row['DEPARTAMENTO']][row['PROVINCIA']][row['AGRUPACION_POLITICA']] = row['VOTOS_OBTENIDOS']

    # Asegurar que todas las provincias tienen todas las agrupaciones con votos default a 0
    for depto, provincias in resultado_final.items():
        for prov, agrupaciones in provincias.items():
            for agrupacion in todas_agrupaciones:
                agrupaciones.setdefault(agrupacion, 0)

    # Convertir a formato deseado
    lista_final = []
    for depto, provincias in resultado_final.items():
        for prov, agrupaciones in provincias.items():
            lista_final.append({
                'DEPARTAMENTO': depto,
                'PROVINCIA': prov,
                'AGRUPACIONES_POLITICAS': agrupaciones
            })

    return jsonify(lista_final)
@app.route('/analisis_distrito', methods=['GET'])
def analisis_distrito():
    provincia = request.args.get('provincia', None)
    datayear = request.args.get('datayear', '2022')

    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022']

    grouped_data_obtenidos = data.groupby(['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']).agg({
        'VOTOS_OBTENIDOS': 'sum',
    }).reset_index()
    data = data.drop_duplicates(subset=['MESA'])

    # Agrupar los datos por departamento, provincia y distrito
    grouped_data = data.groupby(['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO']).agg({
        'ELECTORES_HABILES': 'sum',
        'VOTOS_BLANCOS': 'sum',
        'VOTOS_NULOS': 'sum',
        'VOTOS_IMPUG': 'sum'
    }).reset_index()

    # Calcular el total de no votantes por distrito
    grouped_data['TOTAL_VOTOS'] = (
        grouped_data['VOTOS_BLANCOS'] +
        grouped_data['VOTOS_NULOS'] +
        grouped_data_obtenidos['VOTOS_OBTENIDOS'] +
        grouped_data['VOTOS_IMPUG']
    )

    grouped_data['NO_VOTANTES'] = grouped_data['ELECTORES_HABILES'] - grouped_data['TOTAL_VOTOS']
    grouped_data['VOTOS_OBTENIDOS_TOTAL'] = grouped_data_obtenidos['VOTOS_OBTENIDOS']

    if provincia:
        grouped_data = grouped_data[grouped_data['PROVINCIA'].str.contains(provincia, case=False, na=False)]

    result = grouped_data.to_dict(orient="records")

    return jsonify(result)

@app.route('/analisis_distrito-politicos', methods=['GET'])
def analisis_distrito_politicos():
    provincia = request.args.get('provincia', None)
    datayear = request.args.get('datayear', '2022')

    # Cargar los datos del año especificado o por defecto 2022
    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022']

    # Filtrar los datos por provincia si se especifica una
    if provincia:
        data = data[data['PROVINCIA'].str.contains(provincia, case=False, na=False)]

    # Agrupar datos por departamento, provincia, distrito y agrupaciones políticas
    data_agrupada = data.groupby(['DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'AGRUPACION_POLITICA'])['VOTOS_OBTENIDOS'].sum().reset_index()

    # Reestructurar datos para incluir todas las agrupaciones conocidas
    resultado_final = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for index, row in data_agrupada.iterrows():
        resultado_final[row['DEPARTAMENTO']][row['PROVINCIA']][row['DISTRITO']][row['AGRUPACION_POLITICA']] = row['VOTOS_OBTENIDOS']

    # Asegurar que todos los distritos tienen todas las agrupaciones con votos default a 0
    for depto, provincias in resultado_final.items():
        for prov, distritos in provincias.items():
            for dist, agrupaciones in distritos.items():
                for agrupacion in todas_agrupaciones:
                    agrupaciones.setdefault(agrupacion, 0)

    # Convertir a formato deseado
    lista_final = []
    for depto, provincias in resultado_final.items():
        for prov, distritos in provincias.items():
            for dist, agrupaciones in distritos.items():
                lista_final.append({
                    'DEPARTAMENTO': depto,
                    'PROVINCIA': prov,
                    'DISTRITO': dist,
                    'AGRUPACIONES_POLITICAS': agrupaciones
                })

    return jsonify(lista_final)
@app.route('/analisis_genero', methods=['GET'])
def analisis_genero():
    limit = int(request.args.get('limit', 10))  # Valor por defecto 10
    offset = int(request.args.get('offset', 0))  # Valor por defecto 0
    provincia = request.args.get('provincia', None)  # Valor por defecto None si no se proporciona
    datayear = request.args.get('datayear', '2022-g')  # Cambié a '2022-g' para usar el archivo correcto

    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022-g']  # Usar datos de género por defecto

    # Agrupar datos por región, provincia, distrito y calcular métricas de género
    grouped_data = data.groupby(['Región', 'Provincia', 'Distrito']).agg({
        'Número de electores': 'sum',
        'Electores varones': 'sum',
        '% Electores varones': 'mean',  # Promedio de porcentajes
        'Electores mujeres': 'sum',
        '% Electores mujeres': 'mean',  # Promedio de porcentajes
        'Electores jóvenes *': 'sum',
        '% Electores jóvenes *': 'mean',  # Promedio de porcentajes
        'Electores mayores de 70 años': 'sum',
        '% Electores mayores de 70 años': 'mean'  # Promedio de porcentajes
    }).reset_index()

    if provincia:
        grouped_data = grouped_data[grouped_data['Provincia'].str.contains(provincia, case=False, na=False)]

    # Convertir el DataFrame a JSON
    limited_data = grouped_data.iloc[offset:offset + limit]
    result = limited_data.to_dict(orient="records")

    return jsonify(result)

@app.route('/analisis_genero_provincia', methods=['GET'])
def analisis_genero_provincia():
    departamento = request.args.get('departamento', None)
    datayear = request.args.get('datayear', '2022-g')

    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022-g']

    # Agrupar los datos por región y provincia
    grouped_data = data.groupby(['Región', 'Provincia']).agg({
        'Número de electores': 'sum',
        'Electores varones': 'sum',
        'Electores mujeres': 'sum',
        'Electores jóvenes *': 'sum',
        'Electores mayores de 70 años': 'sum'
    }).reset_index()

    # Calcular porcentajes de género y de grupos de edad
    grouped_data['% Electores varones'] = grouped_data['Electores varones'] / grouped_data['Número de electores']
    grouped_data['% Electores mujeres'] = grouped_data['Electores mujeres'] / grouped_data['Número de electores']
    grouped_data['% Electores jóvenes'] = grouped_data['Electores jóvenes *'] / grouped_data['Número de electores']
    grouped_data['% Electores mayores de 70 años'] = grouped_data['Electores mayores de 70 años'] / grouped_data['Número de electores']
    grouped_data['Electores adultos'] = grouped_data['Número de electores'] - grouped_data['Electores jóvenes *'] + grouped_data['Electores mayores de 70 años']
    if departamento:
        grouped_data = grouped_data[grouped_data['Región'].str.contains(departamento, case=False, na=False)]

    result = grouped_data.to_dict(orient="records")

    return jsonify(result)
@app.route('/analisis_distrito_genero', methods=['GET'])
def analisis_distrito_genero():
    provincia = request.args.get('provincia', None)
    datayear = request.args.get('datayear', '2022-g')  # Asumiendo '2022-g' es la clave para los datos de género
    
    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022-g']

    # Filtrar los datos por provincia si se especifica
    if provincia:
        data = data[data['Provincia'].str.contains(provincia, case=False, na=False)]

    # Agrupar datos por departamento, provincia y distrito para estadísticas de género
    grouped_data = data.groupby(['Región', 'Provincia', 'Distrito']).agg({
        'Número de electores': 'sum',
        'Electores varones': 'sum',
        'Electores mujeres': 'sum',
        'Electores jóvenes *': 'sum',
        'Electores mayores de 70 años': 'sum'
    }).reset_index()

    # Calcular porcentajes de electores varones y mujeres
    grouped_data['% Varones'] = grouped_data['Electores varones'] / grouped_data['Número de electores'] * 100
    grouped_data['% Mujeres'] = grouped_data['Electores mujeres'] / grouped_data['Número de electores'] * 100
    grouped_data['Electores adultos'] = grouped_data['Número de electores'] - grouped_data['Electores jóvenes *'] + grouped_data['Electores mayores de 70 años']

    # Convertir el DataFrame a JSON
    result = grouped_data.to_dict(orient="records")

    return jsonify(result)




@app.route('/analisis_resultados', methods=['GET'])
def analisis_resultados_provincia():
    departamento = request.args.get('departamento', None)
    datayear = request.args.get('datayear', '2022-r')

    # Cargar los datos del año especificado o por defecto 2022
    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022-r']

    # Filtrar los datos por departamento si se especifica uno
    if departamento:
        data = data[data['Región'].str.contains(departamento, case=False, na=False)]

    # Agrupar datos por Región y Organización Política
    data_agrupada = data.groupby(['Región', 'Organización Política', 'Tipo Organización Política']).agg({
        'Electores': 'first',
        'Participación': 'first',
        '% Participación': 'first',
        'Ausentismo': 'first',
        '% Ausentismo': 'first',
        'Votos emitidos': 'first',
        '% Votos emitidos': 'first',
        'Votos válidos': 'first',
        '% Votos válidos': 'first',
        'Votos': 'sum',  # Sumar los votos
        '% Votos': 'first'
    }).reset_index()

    # Reestructurar datos para incluir todas las organizaciones conocidas
    todas_organizaciones = data['Organización Política'].unique()
    resultado_final = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for index, row in data_agrupada.iterrows():
        resultado_final[row['Región']][row['Organización Política']]['Tipo Organización Política'] = row['Tipo Organización Política']
        resultado_final[row['Región']][row['Organización Política']]['Votos'] = row['Votos']

        resultado_final[row['Región']][row['Organización Política']]['% Votos'] = row['% Votos']

    # Asegurar que todas las regiones tienen todas las organizaciones con votos default a 0
    for region in resultado_final.keys():
        for organizacion in todas_organizaciones:
            if organizacion not in resultado_final[region]:
                resultado_final[region][organizacion] = {
                    'Tipo Organización Política': '',
                    'Votos': 0,
                    '% Votos': 0
                }

    # Convertir a formato deseado
    lista_final = []
    for region, organizaciones in resultado_final.items():
        for organizacion, datos in organizaciones.items():
            lista_final.append({
                'Región': region,
                'Organización Política': organizacion,
                'Tipo Organización Política': datos['Tipo Organización Política'],
                'Votos': datos['Votos'],
                '% Votos': datos['% Votos']
            })

    return jsonify(lista_final)









def preparar_datos_para_prediccion(data):
    # Agrupar datos por mesa y partido y sumar los votos
    grouped_data = data.groupby(['UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'MESA', 'AGRUPACION_POLITICA']).agg({
        'VOTOS_OBTENIDOS': 'sum',
        'ELECTORES_HABILES': 'first',
        'VOTOS_BLANCOS': 'first',
        'VOTOS_NULOS': 'first',
        'VOTOS_IMPUG': 'first'
    }).reset_index()

    # Calcular la suma total de votos por cada mesa
    grouped_data['TOTAL_VOTOS'] = (grouped_data['VOTOS_OBTENIDOS'] +
                                   grouped_data['VOTOS_BLANCOS'] +
                                   grouped_data['VOTOS_NULOS'] +
                                   grouped_data['VOTOS_IMPUG'])

    # Calcular el número de personas que no votaron
    grouped_data['NO_VOTANTES'] = grouped_data['ELECTORES_HABILES'] - grouped_data['TOTAL_VOTOS']

    # Añadir una columna para verificar si la suma de votos coincide con los electores habilitados
    grouped_data['VERIFICACION'] = (grouped_data['TOTAL_VOTOS'] + grouped_data['NO_VOTANTES']) == grouped_data['ELECTORES_HABILES']

    # Definir X (características) e y (variable objetivo)
    X = grouped_data[['ELECTORES_HABILES', 'VOTOS_BLANCOS', 'VOTOS_NULOS', 'VOTOS_IMPUG']]
    y = (grouped_data['VOTOS_OBTENIDOS'] / grouped_data['ELECTORES_HABILES'] > 0.5).astype(int)
    return X, y

@app.route('/prediccion_participacion', methods=['GET'])
def prediccion_participacion():
    datayear = request.args.get('datayear', '2022')

    if datayear in data_files:
        data = data_files[datayear]
    else:
        data = data_files['2022']
    
    X, y = preparar_datos_para_prediccion(data)

    # Balancear las clases usando SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    # Dividir datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
    
    # Definir el modelo de Random Forest
    modelo = RandomForestClassifier(random_state=42)
    
    # Definir la búsqueda de hiperparámetros
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Realizar la búsqueda de hiperparámetros con validación cruzada
    grid_search = GridSearchCV(estimator=modelo, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    mejor_modelo = grid_search.best_estimator_
    
    # Realizar predicciones y calcular precisión
    y_pred = mejor_modelo.predict(X_test)
    precision = accuracy_score(y_test, y_pred)
    
    # Preparar resultados
    resultado = {
        'precision': precision,
        'predicciones': y_pred.tolist(),
        'informe_clasificacion': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return jsonify(resultado)


if __name__ == '__main__':
    app.run(debug=True)
