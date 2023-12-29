import random, time

NUMERO_NEURONIOS = 128

def word2vec(string):
    return [ord(c) for c in string] # TODO adicionar nao-linearidade a isso

def vec2word(vec):
    string = ""
    for valor in vec:
        valor_limitado = min(0x10ffff, max(0, int(valor)))
        # ignora pares substitutos
        if 0xD800 <= valor_limitado <= 0xDFFF:
            continue
        try:
            string += chr(int(valor_limitado))
        except ValueError:
            pass
    return string

def relu(x):
    return max(0, x)

def func_ativacao(entrada, peso):
    soma_ponderada = sum(e * p for e, p in zip(entrada, peso))
    return relu(soma_ponderada)

def func_erro(saida_real, saida_esperada):
    erro_quadratico = 0

    for real, esperado in zip(saida_real, saida_esperada):
        diferenca = real - esperado
        erro_quadratico += diferenca ** 2

    return erro_quadratico


class Neuronio:
    def __init__(self, id):
        self.id = id
        self.resultado_ultima_execucao = 0
        self.peso = [random.randint(0, 3), 0]
        
    def execute(self, entrada):
        self.resultado_ultima_execucao = func_ativacao(entrada, self.peso)
        return self.resultado_ultima_execucao    

def resultado_final_neuronios(vetor_resultados):
    vetor_resultante = []    
    for resultado in vetor_resultados:
        tamanho_add_vetor = int(max(1, resultado // 5)) % 8
        for n in range(0, tamanho_add_vetor):
            vetor_resultante.append(resultado)
    return vetor_resultante

neuronios = [Neuronio(i) for i in range(1, NUMERO_NEURONIOS)]

entrada = word2vec("Olá, tudo bem?")
saida_real = 0
saida_esperada = word2vec("Oi!")

vetor_resultados = []
for neuronio in neuronios:
    neuronio.execute(entrada)
    vetor_resultados.append(neuronio.resultado_ultima_execucao)

saida_real = resultado_final_neuronios(vetor_resultados)
print("\n\n====BACKPROPAGATION====\n\n")


def aplicar_rede_neural(entrada):
    vetor_resultados = []
    for neuronio in neuronios:
        vetor_resultados.append(neuronio.execute(entrada))
    return resultado_final_neuronios(vetor_resultados)

taxa_aprendizado = 0.0001

def backpropagation(entrada, saida_real, saida_esperada, resultados_iniciais):
    global taxa_aprendizado
    while True:
        for neuronio in neuronios:
            saida_antes = aplicar_rede_neural(entrada)
            func_erro_antes = func_erro(saida_antes, saida_esperada)

            # precisamos agora mudar UM POUQUINHO o peso e ver se a função de erro em relação à mudança apenas dele
            # aumentou ou diminuiu
            incremento = 0.1
            neuronio.peso[0] += incremento
            saida_depois = aplicar_rede_neural(entrada)
            func_erro_depois = func_erro(saida_depois, saida_esperada)

            gradiente = (func_erro_depois - func_erro_antes) / incremento
            neuronio.peso[0] -= taxa_aprendizado * gradiente
            #print(neuronios[0].peso[0])
        if random.randint(0, 50) == 1:
            print(vec2word(aplicar_rede_neural(entrada)))


backpropagation(entrada, saida_real, saida_esperada, vetor_resultados)
