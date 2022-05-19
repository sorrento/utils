from u_base import make_folder, inicia, tardado, get_now

from utils import lee_pagina


def google_search(query, filetype=None, only_webpages=False):
    """
Obtiene las links de una búsqueda en Google
https://pypi.org/project/googlesearch-python/
    :param query: consulta  google
    :type filetype: pdf, xls, etc
    :param only_webpages: quita pdfs y excels de la búsqueda
    return: lista con los enlaces
    """
    try:
        from googlesearch import search
    except ImportError:
        print("No module named 'google' found")

    t = inicia('Google search')

    if filetype is not None and only_webpages:
        print('ERROR: no debe poner only_webpages y proporcionar filetypes')
        return None

    if only_webpages:
        print('** Sólo webpages.')
        query = query + ' -filetype:pdf -filetype:xls -filetype:xlsx'

    if filetype is not None:
        print('** Sólo archivos tipo .' + filetype)
        query = query + ' filetype:' + filetype

    links = []
    print(query)
    rr = search(query, tld="co.in",  # top level domain
                num=10,
                stop=10,
                pause=2)

    for j in rr:
        # t1 = inicia('uno')
        links.append(j)
        print(j)
        # tardado(t1)

    tardado(t)
    d = {'time':          get_now(),
         'query':         query,
         'only_webpages': only_webpages,
         'filetype':      filetype,
         'links':         links
         }

    return d


def get_country_from_url_code(country_code):
    import pandas as pd
    codes = pd.read_csv('data_in/coutries_codes_url.csv', sep=';')
    code_ = codes[codes.code == country_code]
    if len(code_) > 0:
        country = code_.country.iloc[0]
    else:
        country = '-'

    return country


def procesa_url(url):
    """
Descompone la url en partes como el pais o el dominio
    :param url:
    :return:
    """
    import re
    partes = re.split('[/?]', url)
    filename = [x for x in partes if '=' not in x][-1]
    partes_file = filename.split('.')
    domain = partes[2]
    country_code = domain.split('.')[-1]
    country = get_country_from_url_code(country_code)

    d = {'filename':      filename,
         'filename_bare': partes_file[0],
         'extension':     partes_file[-1],
         'domain':        domain,
         'country_code':  country_code,
         'country':       country}

    return d


def save_file_from_url(url, path, filename=None):
    import urllib.request
    import os

    if not os.path.exists(path):
        make_folder(path)

    if filename is None:
        filename = procesa_url(url)['filename']
        print('Usaaremos el nombre de fichero de la url:', filename)
    path_filename = path + '/' + filename
    print('** guardando el fichero en:', path_filename)
    path, ob = urllib.request.urlretrieve(url, path_filename)

    ok = True  # todo poner si ha logrado traer el objeto ( cod 200?)
    return ok


def scrap_news(q, scrap_pages=True):
    """
A partir de una consulta de google en news, nos traemos el contenido de las noticias
Utilizamo un servicio de pago, que tienen un plan gratuito https://serpapi.com/dashboard

    :param q: palabras para la consulta
    :param scrap_pages:
    :return un json con la descripción de la consulta y cada resultado. En "body" ponemos
    el resultado de scrapear el texto de la página al meternos
    """
    from serpapi import GoogleSearch
    pk = 'd77f9c9f9fc647c1e3881dd62b9e5431bd2dddf2d1f8ba6ae489980cf5ae2782'

    print('** Utilizando el servicio de pago de https://serpapi.com')

    params = {
        "engine":        "google",
        "q":             q,
        "location":      "Austin, Texas, United States",
        "google_domain": "google.com",
        "gl":            "us",
        "hl":            "en",
        "num":           "200",  # las noticias tienen como máximo 100 al parecer
        "tbm":           "nws",
        "api_key":       pk
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    n = len(results['news_results'])
    print('** Se han traído {} resultados'.format(str(n)))

    if scrap_pages:
        for i in range(n):
            base = results['news_results'][i]
            print('\n******* ' + str(base['position']), '  ', base['title'])
            link = base['link']
            print(link)

            if 'body' not in base:
                body = lee_texto_de_website(link)['body']
                base['body'] = body.replace('\n', ' ')
                print('   listo')
            else:
                print('*** ya tiene body')
    else:
        print('** No scrapeamos las páginas. Para eso poner scrap_paages=True')

    return results


def lee_texto_de_website(url):
    """
trae el texto de una página web
    :param url:
    :return: diccionario con el titulo y el cuerpo de la página
    """
    doc, html_txt, cuerpo = lee_pagina(url)
    j = {'title': doc.title(), 'body': cuerpo}
    return j
