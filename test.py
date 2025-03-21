

def read_first_100_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= 100:
                    break
                print(line.strip())
    except FileNotFoundError:
        print(f"Fișierul nu a fost găsit: {file_path}")
    except Exception as e:
        print(f"Eroare la citirea fișierului: {e}")


file_path = input("Introduceți calea absolută a fișierului: ")
read_first_100_lines(file_path)
