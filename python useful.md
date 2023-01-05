# Jupyter

    from platform import python_version
    python_version ()

Autorecarga las dependencias en cada celda

    %load_ext autoreload
    %autoreload 2

Para que cargue sin errores los `display`

    from IPython.core.display import display

Usar todo el ancho de la pantalla

    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:100% !important; }</style>"))


## 1. Sincronización
 Para que sincronice un fichero `.py` con el `.pynb`, poner esto en el metadata

    "jupytext": {
    "formats": "ipynb,py"
    },

## 2. Nuevo kernel en jupyter
Escribir esto en la consola de anaconda

`ipython kernel install --name “data-science” --user`

# Editable dataframes (interactivos)

    import ipysheet
    sheet = ipysheet.from_dataframe(df);sheet
    df=ipysheet.to_dataframe(sheet) #recuperar el valor

# ejemplos de hilos

    import threading
    def princ(hilo):
        from u_base import now
        for i in range(5):
            print(hilo, now())
            time.sleep(1)
    
        return hilo
    t1 = threading.Thread(target=princ, args=('A'))
    t2 = threading.Thread(target=princ, args=('B'))
    threads = [t1, t2]
    t1.start()
    t2.start()
    t1.join()
    t2.join()

# MATPLOTLIB

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(b.DATE, b.Value,'.')
    plt.plot(b.DATE, b.Value)
    plt.ylim(0)
    plt.xlabel('date')
    plt.ylabel('Index price')
    plt.title(i)
    plt.show()

## Subplots
    from matplotlib.pyplot import figure
    fig, axs = plt.subplots(2, 2,figsize=(12, 8), dpi=80)
    
    axs[0,0].plot(total_norm.yw_start_fut, total_norm.Value_log_fut)
    
    axs[1,0].plot(total_norm.yw_start, total_norm[w].rolling(r).mean())
    axs[1,0].set_title(f'word:{w} ROLLING MEAN:{r} sems')
    
    axs[0,1].plot(np.log(total_norm.Value_fut), total_norm[w].rolling(r).mean(), 'o')


# PLOTLY
Evitar que desaparezcan los plots:
    
    import plotly.io as pio
    pio.renderers.default = 'notebook'  # para que no desaparezcan los plots

Típico plot:

    import plotly.express as px
    fig = px.line(df, x="nfeat", y=aa, text="nfeat", color='motor',
                  hover_name='nfeat',
                  #               hover_data=[S],
                  hover_data={aa: ':.4f', 'motor': False, 'nfeat': False, S: True},
                  title=titulo)
    fig.update_traces(textposition="bottom right")
# CUDA
installing pytorch
    https://blog.machinfy.com/installing-pytorch/

    torch.cuda.is_available()

# Pandas

## Crear columna desde dos
    df.apply(lambda row: time_from_quarter(row.year, row.quarter), axis=1)



# Otros
## Decorators
por ejemplo para hacer imprimir los tiempos de cada funcion
    https://pythongeeks.org/python-decorators/

