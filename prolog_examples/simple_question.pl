dog(fido).
mammal(X) :- dog(X).
spider(shelob).
fish(billy).
swim(X) :- fish(X).
swim(X) :- dog(X).
