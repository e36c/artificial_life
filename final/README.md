# ME495 Artificial Life - Final project
*Charlie Seifert, Winter 2025*

I have departed from my previous work in this class to create an entirely new creature morphology and simulation for this final project, described in detail below.

## Creature Morphology
The creature is made up of four mpm regions: A passive (unactuated) center carapace of set width and height, and four "fins", one on each side of the carapace. The fins are defined by three parameters:
- position
- length
- width

The position represents the location of the center of the fin along the edge it belogns to, and is represented by a number in the range [0,1], with 0 being closest to the adjoining counterclockwise edge, and 1 being closest to the adjoining clockwise edge. Moving beyond [0,1] is possible, but not recommended; the fin is likely to fall off during the course of the simulatin. The width and length can be any number. The length represents the dimension extending out from carapace, and the width is the other dimension. The construction of this creature is handled by the ```robot()``` function. All of its features are rectangles.

## Evolutionary loop
A pair of nested loops makes up the evolutionary loop for our four-finned creature.
### Inner Loop
The inner loop trains the creature via gradient descent to walk to the right of the screen. It is very similar to that used in the original diffmpm.py difftaichi example. Each iteration of the outer loop runs 15 inner loops by default, as I found loss values tended to not change much after that point, and does so with a single morphology to find the best strategy for the creature to move with that morphology.
### Outer Loop
The first time through, the outer loop just saves the loss and fin data, generates a mutation, and jumps to the next 15 inner loops with the newly mutated creature. Subsequent outer loops compare the final loss value of a morphology with that of the previous creature. If it has improved, it will allow the just-tested mutation to propogate into the next generation, as well as add another mutation. If it has not improved, it will revert the creature back to its "parent" (effectively killing off the bad mutation), apply another random mutation to the parent, then run again.

## Taichi Simulation and Program Structure
This program basically wraps the core diffmpm.py simulation with an outer morphology loop that selects for better-performing morphology via competition with the parent. The cost function is just the distance from the target; the only pressure on morphology is enforced by the outer loop. I found that the simplest way to re-run the simulation every time was to reset Taichi every iteration of outer loop, avoiding a problem I faced before where taichi was unhappy reallocating the fields. Becuase taichi is reset every time, the program handles the propogation of the values most relevant to the creature in normal Python data structures.