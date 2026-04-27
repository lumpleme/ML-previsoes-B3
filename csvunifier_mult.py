from pathlib import Path
import xlrd
import csv
from datetime import datetime



def float_ou_none(valor):
	"""Coloca vazio como None ou converte o dado numerico para float."""
	if valor == "":
		return None
	try:
		# Remove pontos e substitui vírgula por ponto
		if isinstance(valor, str):
			valor = valor.replace(".", "").replace(",", ".")
		valor = float(valor)*10**12
		return valor
	except (TypeError, ValueError):
		return valor


def extrair_historico_ibovespa_xls(caminho_xls):
	"""Percorre todas as abas do XLS e monta os registros em dicionarios.

	Regras aplicadas:
	- ano: nome da aba
	- mes: B/Jan=1, C/Fev=2, ..., M/Dez=12
	- dia: valor da coluna A (linhas 3 a 33)
	- valor: valor da celula (B3:M33)
	"""
	arq = Path(caminho_xls)
	if not arq.exists():
		raise FileNotFoundError(f"Arquivo nao encontrado: {arq}")

	workbook = xlrd.open_workbook(str(arq))
	dados = []

	for aba in workbook.sheets():
		ano = int(aba.name)

		# Linhas 3..33 no Excel => indices 2..32.
		for l in range(2, 33):
			try:
				dia = int(aba.cell_value(l, 0))
			except (ValueError, TypeError):
				continue

			# Colunas B..M no Excel => indices 1..12.
			for col in range(1, 13):
				mes = int(col)  # B=1, C=2, ..., M=12
				valor = float_ou_none(aba.cell_value(l, col))

				dados.append(
					{
						"ano": ano,
						"mês": mes,
						"dia": dia,
						"valor": valor,
					}
				)

	return dados

def adicionar_historico_ibovespa_csv(nome_csv, historico):
	"""Adiciona o histórico do ano em CSV ao dicionário passado."""
	with open(nome_csv, 'r', encoding='latin-1') as arq:
		# Pega o ano da linha 1 "IBOVESPA - AAAA"
		ano = int(arq.readline().split()[-1])
		
		# Pula linha 2 (header)
		arq.readline()
		# Pula linha 3 (em branco)
		arq.readline()
		
		# Processa dias
		for linha in arq:
			linha = linha.strip()
			if not linha or linha.startswith('M'):  # Ignora linhas vazias e MÍNIMO/MÁXIMO
				continue
			
			valores = linha.split(';')
			
			try:
				dia = int(valores[0])
			except (ValueError, IndexError):
				continue
			
			# Processa cada mês (Jan=1, Fev=2, ..., Dez=12)
			for mes_idx in range(1, 13):
				if mes_idx < len(valores):
					valor = float_ou_none(valores[mes_idx])
					
					historico.append({
						"ano": ano,
						"mês": int(mes_idx),
						"dia": dia,
						"valor": valor
					})

def historico_para_csv(nome_csv, historico):
	"""Salva o histórico em um arquivo CSV com formato data(AAAA-MM-DD),valor."""
	# Ordena cronologicamente por ano, mês e dia
	historico_ordenado = sorted(historico, key=lambda x: (x['ano'], x['mês'], x['dia']))
	
	with open(nome_csv, 'w', newline='', encoding='latin-1') as arq:
		writer = csv.DictWriter(arq, fieldnames=["data", "valor"])
		writer.writeheader()
		for registro in historico_ordenado:
			# Formata a data como AAAA-MM-DD
			data = f"{int(registro['ano'])}-{int(registro['mês']):02d}-{int(registro['dia']):02d}"
			writer.writerow({"data": data, "valor": registro['valor']})

if __name__ == "__main__":
	arquivo = "IBOVDIA.XLS"
	historico_ibovespa = extrair_historico_ibovespa_xls(arquivo)
	adicionar_historico_ibovespa_csv("Evolucao_Diaria.csv", historico_ibovespa)
	for i in range(1, 29):
		nome = f"Evolucao_Diaria ({i}).csv"
		adicionar_historico_ibovespa_csv(nome, historico_ibovespa)

	historico_para_csv("Historico_IBovespa_Revertido.csv", historico_ibovespa)

	print(f"Total de registros: {len(historico_ibovespa)}")
	print(f"Primeiros 5 registros:")
	for registro in historico_ibovespa[:5]:
		print(registro)

	print("Últimos 5 registros:")
	for registro in historico_ibovespa[-5:]:
		print(registro)
