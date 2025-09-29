# Types of Stochascity

In STORI, stochasticity types 1, 2.1, 2.2, and 3.2 are implemented as extensions of Type 3.1 environments. This is because screen-based observations serve as the default, well-studied ALE inputs for various reinforcement learning algorithms, providing a consistent foundation for comparing different types of stochasticity while also allowing for interpretable analysis of agent actions and behaviors.

## Type 0
This type returns the RAM state of the game (a 1-D numpy array) with state labels as the observation. (**Work In Progress**)

## Type 1
The ‘ActionDependentStochasticityWrapper‘ randomly replaces the agent’s intended action with a random action from the action space with a specified probability.

## Type 2.1
The ‘ActionIndependentRandomStochasticityWrapper‘ implements environment specific random events that occur independently of the agent’s actions. These effects are applied probabilistically and create unpredictable environmental changes to which the agent must adapt.

## Type 2.2
This introduces temporal concept drift where the environment dynamics change over time. The ‘ActionIndependentConceptDriftWrapper‘ supports both sudden and cyclic modes between 2 concepts. The concept 1 is the default environment (type 3.1) and concept 2 can be any other environment stochasticity types out of 1, 2.1 and 3.2. In sudden mode, the environment switches to concept 2 after a fixed number of steps. In cyclic mode, it alternates between the concept 1 and 2 every specified number of steps, creating a challenging environment where the agent must continuously adapt to changing dynamics.

## Types 3.1
This stochasticity type returns the default ALE environment without any modifications.

## Types 3.2
The ‘PartialObservationWrapper‘ introduces partial observability by modifying the agent’s observations. The system supports multiple observation modification techniques including cropping (removing portions of the screen), blackout (hiding specific regions), and RAM manipulation (temporarily modifying the game’s internal state to get modified observation).
