from typing import List, Dict
import random


def crossover_keywords(keywords1: List[str], keywords2: List[str]) -> List[str]:
    """
    Realiza crossover entre dos arrays de keywords.
    Intercambia elementos entre los arrays de manera aleatoria.
    """
    set1 = set(keywords1)
    set2 = set(keywords2)

    unique_to_1 = set1 - set2
    unique_to_2 = set2 - set1

    # Si ambos están vacíos → devuelve la unión
    if not unique_to_1 and not unique_to_2:
        return list(set1 | set2)

    # Manejo seguro para evitar randint(1,0)
    num_to_swap_1 = random.randint(1, len(unique_to_1)) if unique_to_1 else 0
    num_to_swap_2 = random.randint(1, len(unique_to_2)) if unique_to_2 else 0

    elements_to_swap_1 = random.sample(list(unique_to_1), num_to_swap_1) if num_to_swap_1 else []
    elements_to_swap_2 = random.sample(list(unique_to_2), num_to_swap_2) if num_to_swap_2 else []

    common_elements = set1 & set2
    new_keywords = list(common_elements | set(elements_to_swap_1) | set(elements_to_swap_2))

    return new_keywords


def crossover(padre1: Dict, padre2: Dict) -> Dict:
    """
    Realiza crossover entre dos padres en parámetros variados.
    Retorna un hijo con características de ambos padres.
    """
    hijo = padre1.copy()

    parametros = ["rol", "topic", "keywords"]

    # Selecciona aleatoriamente entre 1 y 3 parámetros a intercambiar
    p = random.randint(1, len(parametros))
    parametros_a_intercambiar = random.sample(parametros, p)

    for parametro in parametros_a_intercambiar:
        if parametro == "rol":
            hijo["rol"] = padre2["rol"]
        elif parametro == "topic":
            hijo["topic"] = padre2["topic"]
        elif parametro == "keywords":
            hijo["keywords"] = crossover_keywords(padre1["keywords"], padre2["keywords"])

    # Resetear campos que dependen de los anteriores
    hijo["prompt"] = ""
    hijo["generated_data"] = ""
    hijo["fitness"] = 0

    return hijo


def crear_individuo_ejemplo(rol: str, topic: str, keywords: List[str], fitness: float = 0.0) -> Dict:
    """Crea un individuo de ejemplo"""
    return {
        "rol": rol,
        "topic": topic,
        "prompt": f"Example prompt for {rol} about {topic}",
        "keywords": keywords,
        "generated_data": "",
        "fitness": fitness,
    }


def mostrar_individuo(individuo: Dict, nombre: str):
    """Muestra la información de un individuo"""
    print(f"\n{nombre}:")
    print(f"  Rol: {individuo['rol']}")
    print(f"  Topic: {individuo['topic']}")
    print(f"  Keywords: {individuo['keywords']}")
    print(f"  Prompt: {individuo['prompt']}")
    print(f"  Generated Data: {individuo['generated_data']}")
    print(f"  Fitness: {individuo['fitness']}")


def test():
    padre1 = crear_individuo_ejemplo(
        rol="a healthcare worker",
        topic="coronavirus pandemic",
        keywords=["covid", "pandemic", "healthcare", "worker", "emergency"],
        fitness=0.75,
    )

    padre2 = crear_individuo_ejemplo(
        rol="a journalist",
        topic="climate change",
        keywords=["climate", "change", "environment", "journalist", "reporting"],
        fitness=0.82,
    )

    print("\nIndividuos:")
    print("=" * 60)
    mostrar_individuo(padre1, "Padre 1")
    mostrar_individuo(padre2, "Padre 2")

    print("\nAplicar Crossover:")
    print("=" * 60)
    hijo = crossover(padre1, padre2)
    mostrar_individuo(hijo, "Hijo")


if __name__ == "__main__":
    test()
