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
    plt.figure(figsize=(9, 5))
    plt.plot(b.DATE, b.Value,'.')
    plt.plot(b.DATE, b.Value)
    plt.ylim(0)
    plt.xlabel('date')
    plt.ylabel('Index price')
    plt.title(i)
    plt.show()

# CUDA
installing pytorch
    https://blog.machinfy.com/installing-pytorch/

    torch.cuda.is_available()

# Pandas

## Crear columna desde dos
    df.apply(lambda row: time_from_quarter(row.year, row.quarter), axis=1)