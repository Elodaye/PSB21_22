# PSB21_22

# Projet de classification sonore par méthodes de machine learning

Reconnaissance automatiques des différents sons présents dans la ville de Brest. 
Méthodes de deep learning, réseau de neurones convolutifs. 
Apprentissage effectué sur les spectrogrammes des sons : représentation de l'amplitude du signal, pour un maillage temps-fréquence. 

La restructuration du code est prévue pour les semaines à venir.


## Consignes pour le groupe

Pour l’instant les classes labellisées sont : car, pie, moto, chien. Si autres, il faut les rajouter dans ListClass et classIndice. Pie = les oiseaux en général.

Il faut bien penser à indexer le self.i du doc n°1 à partir du numéro de recording suivant, pour le moment c’est manuel. (si on fait chacun notre tour, la variable serait encore  plus globale qu’une variable système, ça me paraît compliqué à automatiser). self.ii part toujours de 0 par contre. 
Actuellement self.i vaut 21 initialement car ma première salve contenait 21 recording de 10 secondes (enfin sur 3 :35 min on pouvait en faire 21), mais pour la suite 
il faut passer à 42, avec ceux que je viens de rajouter. Quand on prendra ton relais on mettra aussi à jour à partir des recording que tu as ajouté. 

Bien penser à changer le path, mais ça peut être un peu problématique, à moins qu’on crée un pareil sur chacun de nos ordis : dans « mes_datas », qui regroupent tout ce qu’on a labellisé, les paths sont de la forme « C:/Users/Utilisateur/Documents/ENSTA/2A/UE 3.4/Projet_système/Machine_learning/Donnees_label/wav/spec/recording27.png ». 
On peut changer les paths de ce qu’on labellise perso, mais pas tant les anciens, du coup si on avait chacun un répertoire du même nom sur nos ordi ça pourrait être plus simple 
(avec un path plus simple que ça d’ailleurs éventuellement).

Si on ne mélange pas du tout (manuellement) l’ordre de ce qu’on a record et que se retrouve dans « mes_datas », parfois des classes ne sont pas testées dans les données de  validation par exemple, et ça c’est pas bon, ça fait erreur de toute façon donc on le voit.
L’actuel « 21 et 13 » à la ligne 42 du code n°2 sont le nb de données en entrainement et validation, et les données de test sont celles qui restent (ici 42 – 21 -13). 
Ces nombres vont donc augmenter (manuellement pour le moment, après why not fixer une proportion et ça évolue tout seul) au fur et à mesure qu’on augmente le nb de valeurs 
labellisées.

Si tu télécharge des audios sur https://lasonotheque.org/search?q=ville l’audio est pas exploitable directement (ex village), il faut l’ouvrir dans audacity, faire « edition »  puis « metadonnées » et « retirer ». Après Valider puis Fichier et Exporter le nouvel audio qui normalement fonctionne (village_2). De plus les données de ce site sont en stéréo (2D), alors qu’avec l’audiomoth elles sont en audio (1D), donc aux lignes 160 et 228 du doc n°2 on extrait la première composante. Selon les données d’entrées il n’y aura pas besoin  du « ,0 ». 
