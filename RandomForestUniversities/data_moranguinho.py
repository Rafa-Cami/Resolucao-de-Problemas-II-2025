
def tratamento_data(data):
    data=str(data)
    data_sem_barra = data.replace("/", "")
    data_invertida = "".join(reversed(data_sem_barra))
    print(data_invertida)


def main():
    print("Escolha a data de partida no formato: dd/mm/aaaa")
    data_partida = input()
    print("Escolha a data de chegada no formato: dd/mm/aaaa")
    data_chegada = input()

    anomesdia_partida = tratamento_data(data_partida)
    anomesdia_chegada = tratamento_data(data_chegada)

main ()