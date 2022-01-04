import csv
from itertools import zip_longest

def write_list_to_file(kernel, resultCompor, resultPsic, resultFisio):
    """Write the list to csv file."""
    resultFile = 'results/resultados.csv'
    rows = [kernel, resultCompor, resultPsic, resultFisio]
    export_data = zip_longest(*rows, fillvalue='')
    with open(resultFile, "w") as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("Kernel", "Resultado baixa", "Resultado média", "Resultado alta"))
        wr.writerows(export_data)
        myfile.close()