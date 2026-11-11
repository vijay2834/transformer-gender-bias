"""Word sets from Caliskan et al. (2017), "Semantics derived automatically from
language corpora contain human-like biases", WEAT tests 6-8, plus definitional
gender pairs from Bolukbasi et al. (2016)."""

MALE_NAMES = ["john", "paul", "mike", "kevin", "steve", "greg", "jeff", "bill"]
FEMALE_NAMES = ["amy", "joan", "lisa", "sarah", "diana", "kate", "ann", "donna"]
MALE_TERMS = ["male", "man", "boy", "brother", "he", "him", "his", "son"]
FEMALE_TERMS = ["female", "woman", "girl", "sister", "she", "her", "hers", "daughter"]

CAREER = ["executive", "management", "professional", "corporation", "salary", "office", "business", "career"]
FAMILY = ["home", "parents", "children", "family", "cousins", "marriage", "wedding", "relatives"]
MATH = ["math", "algebra", "geometry", "calculus", "equations", "computation", "numbers", "addition"]
ARTS = ["poetry", "art", "dance", "literature", "novel", "symphony", "drama", "sculpture"]
SCIENCE = ["science", "technology", "physics", "chemistry", "einstein", "nasa", "experiment", "astronomy"]
ARTS_2 = ["poetry", "art", "shakespeare", "dance", "literature", "novel", "symphony", "drama"]

WEAT_TESTS = {
    "WEAT 6: career vs family (names)": (CAREER, FAMILY, MALE_NAMES, FEMALE_NAMES),
    "WEAT 7: math vs arts (terms)": (MATH, ARTS, MALE_TERMS, FEMALE_TERMS),
    "WEAT 8: science vs arts (terms)": (SCIENCE, ARTS_2, MALE_TERMS, FEMALE_TERMS),
}

DEFINITIONAL_PAIRS = [("woman", "man"), ("girl", "boy"), ("she", "he"), ("mother", "father"),
                      ("daughter", "son"), ("gal", "guy"), ("female", "male"), ("her", "his"),
                      ("herself", "himself"), ("mary", "john")]
EQUALIZE_PAIRS = [("woman", "man"), ("girl", "boy"), ("she", "he"), ("mother", "father"), ("daughter", "son"),
                  ("sister", "brother"), ("queen", "king"), ("wife", "husband"), ("female", "male"),
                  ("her", "his"), ("herself", "himself"), ("girls", "boys"), ("women", "men"), ("aunt", "uncle")]
