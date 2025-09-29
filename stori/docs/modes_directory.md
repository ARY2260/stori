# All Implemented Stochasticity Modes
We define stochasticity modes along four Atari environments (Breakout, Boxing, Gopher, BankHeist), and the set of cropping modes are common to all games.

## Common Partial observation CROPPING modes (all games)

- Mode 0: No crop
- Mode 1: Left — Crop the left half of the observation
- Mode 2: Right — Crop the right half of the observation
- Mode 3: Top — Crop the top half of the observation
- Mode 4: Bottom — Crop the bottom half of the observation
- Mode 5: Random circular mask — Randomly mask a circular region of the observation

## BREAKOUT
### Action-independent random
- 0: none
- 1: block hit cancel (reward unchanged)
- 2: block hit cancel (reward set to 0)
- 3: regenerate a randomly chosen hit block
### Partial observation (blackout)
- 0: none
- 1: all
- 2: blocks
- 3: paddle
- 4: score
- 5: ball missing top
- 6: ball missing middle
- 7: ball missing bottom
- 8: blocks and paddle
- 9: blocks and score
- 10: ball missing top and bottom
- 11: ball missing all
### Partial observation - RAM modification
- 0: none
- 1: nus pattern (blocks RAM)
- 2: ball hidden

## BOXING
### Action-independent random
- 0: none
- 1: colorflip (swap player/enemy colors)
- 2: hit cancel (revert score; reward set to 0)
- 3: displace to corners (swap player/enemy positions)
### Partial observation (blackout)
- 0: none
- 1: all
- 2: left boxing ring
- 3: right boxing ring
- 4: full boxing ring
- 5: enemy score
- 6: player score
- 7: enemy+player score
- 8: clock
- 9: enemy+player score+clock
### Partial observation - RAM modification
- 0: none
- 1: hide boxing ring
- 2: hide enemy
- 3: hide player

## GOPHER
### Action-independent random
- 0: none
- 1: hole doesn’t close (fill cancel; reward unchanged)
- 2: hole doesn’t close (fill cancel; reward set to 0)
- 3: randomly remove one visible carrot (once per reset)
### Partial observation (blackout)
- 0: none
- 1: all
- 2: gopher attack (both sides)
- 3: left gopher attack
- 4: right gopher attack
- 5: underground full (before-dug color)
- 6: underground full offset (before-dug color)
- 7: underground row 0 (before-dug)
- 8: underground row 0 (dug color)
- 9: underground row 1 (before-dug)
- 10: underground row 1 (dug color)
- 11: underground row 2 (before-dug)
- 12: underground row 2 (dug color)
- 13: underground row 3 (before-dug)
- 14: underground row 3 (dug color)
- 15: farmer (full)
- 16: farmer below nose
- 17: duck fly
- 18: score
### Partial observation - RAM modification
- 0: none
- 1: hide left carrot
- 2: hide middle carrot
- 3: hide right carrot
- 4: hide all carrots
- 5: hide seed

## BANKHEIST
### Action-independent random
- 0: none
- 1: dropped bomb is a dud
- 2: fuel leaks (per city, once per episode)
- 3: switch city mid-way (teleport)
- 4: bank empty (reward suppressed when bank→police transition detected)
### Partial observation (blackout)
- 0: none
- 1: all
- 2: city walls (all)
- 3: top city wall
- 4: left city wall
- 5: bottom city wall
- 6: right city wall
- 7: left and right city walls together
- 8: fuel region
- 9: lives region
- 10: score region
### Partial observation - RAM modification
- 0: none
- 1: hide robber’s car
- 2: hide change in fuel (always full)
- 3: hide city blocks
- 4: blend city blocks and wall (background color)
- 5: hide banks (when currently a bank)
- 6: hide police (when currently police)

### Concept Drift Usage
All partial observation, action-independent, and action-dependent modes can also be used as a second concept in a concept drift setting, enabling controlled evaluation of robustness to non-stationary environments.
