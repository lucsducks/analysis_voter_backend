import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
data = pd.read_excel('2022/ERM2022_Actas.xlsx')  # Ajusta la ruta según tus datos

# Agrupar datos por mesa y sumar/agregar apropiadamente
grouped_data = data.groupby(['UBIGEO', 'DEPARTAMENTO', 'PROVINCIA', 'DISTRITO', 'MESA']).agg({
    'VOTOS_OBTENIDOS': 'sum',  # Suma de todos los votos obtenidos por las agrupaciones en una mesa
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

# Mostrar las mesas con resultados
print(grouped_data[['MESA', 'TOTAL_VOTOS', 'NO_VOTANTES', 'ELECTORES_HABILES', 'VERIFICACION']].head())

# Opcional: Mostrar inconsistencias si hay alguna
inconsistencias = grouped_data[grouped_data['VERIFICACION'] == False]
print("Inconsistencias encontradas:", len(inconsistencias))
print(inconsistencias)

# Histograma de No Votantes
plt.figure(figsize=(10, 6))
sns.histplot(grouped_data['NO_VOTANTES'], bins=30, kde=True)
plt.title('Distribución de No Votantes por Mesa')
plt.xlabel('Número de No Votantes')
plt.ylabel('Frecuencia')
plt.show()

# Gráfico de barras apiladas para la composición de votos por mesa
grouped_data['TOTAL_VOTOS'] = grouped_data['VOTOS_OBTENIDOS'] + grouped_data['VOTOS_BLANCOS'] + grouped_data['VOTOS_NULOS'] + grouped_data['VOTOS_IMPUG']
grouped_data_short = grouped_data.head(100)  # Limitar a 30 mesas para visualización

# Gráfico de barras apiladas
grouped_data_short.set_index('MESA')[['VOTOS_OBTENIDOS', 'VOTOS_BLANCOS', 'VOTOS_NULOS', 'VOTOS_IMPUG']].plot(kind='bar', stacked=True, figsize=(14, 7))
plt.title('Composición de Votos por Mesa')
plt.xlabel('Mesa')
plt.ylabel('Número de Votos')
plt.legend(title='Tipo de Voto')
plt.show()

# Boxplot para explorar las diferencias en los votos totales vs. electores habilitados
plt.figure(figsize=(10, 6))
sns.boxplot(x='VERIFICACION', y='NO_VOTANTES', data=grouped_data)
plt.title('Distribución de No Votantes por Verificación de Totales')
plt.xlabel('Verificación Correcta')
plt.ylabel('Número de No Votantes')
plt.show()
