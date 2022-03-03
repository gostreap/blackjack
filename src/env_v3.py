from typing import Optional
import os

import numpy as np
import pygame

import gym
from gym import spaces
from gym.utils import seeding


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card(np_random):
    return int(np_random.choice(deck))


def draw_hand(np_random):
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 if (1 in hand and sum(hand) + 10 <= 21) else 0


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


def can_double_down(hand):
    return 1 if len(hand) == 2 else 0


def can_split(hand0, hand1):
    if len(hand1) == 0 and len(hand0) == 2 and hand0[0] == hand0[1]:
        return 1
    else:
        return 0


class BlackjackDoubleDownSplitEnv(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description
    Card Values:

    - Face cards (Jack, Queen, King) have a point value of 10.
    - Aces can either count as 11 (called a 'usable ace') or 1.
    - Numerical cards (2-9) have a value equal to their number.

    This game is played with an infinite deck (or with replacement).
    The game starts with the dealer having one face up and one face down card,
    while the player has two face up cards.

    The player can request additional cards (hit, action=1) until they decide to stop (stick, action=0)
    or exceed 21 (bust, immediate loss).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust, the player wins.
    If neither the player nor the dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.

    ### Action Space
    There are two actions: stick (0), hit (1), double down (2), split (3).

    ### Observation Space
    The observation consists of a 3-tuple containing: the player's current sum,
    the value of the dealer's one showing card (1-10 where 1 is ace),
    and whether the player holds a usable ace (0 or 1).

    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (http://incompleteideas.net/book/the-book-2nd.html).

    ### Rewards
    - win game: +1
    - lose game: -1
    - draw game: 0
    - win game with natural blackjack:

        +1.5 (if <a href="#nat">natural</a> is True)

        +1 (if <a href="#nat">natural</a> is False)

    ### Arguments

    ```
    gym.make('Blackjack-v1', natural=False)
    ```

    <a id="nat">`natural`</a>: Whether to give an additional reward for
    starting with a natural blackjack, i.e. starting with an ace and ten (sum is 21).

    ### Version History
    * v0: Initial versions release (1.0.0)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, natural=False, sab=False):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(32), # sum of player hand 1
                spaces.Discrete(11), # dealer card
                spaces.Discrete(2),  # usable ace hand 1
                spaces.Discrete(2),  # can double down hand 1
                spaces.Discrete(32), # sum of player hand 2
                spaces.Discrete(2),  # usable ace hand 2
                spaces.Discrete(2),  # can double down hand 2
                spaces.Discrete(2),  # can split
                spaces.Discrete(2),  # hand to play
            )
        )

        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural

        # Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
        self.sab = sab

    def has_split(self):
        return len(self.hand1) != 0

    def is_done(self):
        return self.hand0_stop and (self.hand1_stop or len(self.hand1) == 0)

    def stop_hand(self, hand_num, reward):
        if hand_num == 0:
            self.hand0_stop = True
            self.final_reward0 = reward
        else:
            self.hand1_stop = True
            self.final_reward1 = reward

    def get_hand_to_play(self, hand_num):
        if hand_num == 0:
            return self.hand0
        else:
            return self.hand1

    def set_hand_to_play(self, current_hand_num):
        if current_hand_num == 0 and not self.hand1_stop and len(self.hand1) != 0:
            self.hand_to_play = 1
        elif current_hand_num == 1 and not self.hand0_stop:
            self.hand_to_play = 0

    def step(self, action):
        assert self.action_space.contains(action)
        if action == 1:  # hit: add a card to players hand and return
            hand = self.get_hand_to_play(self.hand_to_play)
            hand.append(draw_card(self.np_random))
            if is_bust(hand):
                reward = -1.0
                self.stop_hand(self.hand_to_play, reward)
            else:
                reward = 0.0

        elif action == 0:  # stick: play out the dealers hand, and score
            hand = self.get_hand_to_play(self.hand_to_play)

            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.np_random))

            reward = cmp(score(hand), score(self.dealer))
            if self.sab and is_natural(hand) and not is_natural(self.dealer):
                # Player automatically wins. Rules consistent with S&B
                reward = 1.0
                self.stop_hand(self.hand_to_play, reward)

            elif not self.sab and self.natural and is_natural(hand) and reward == 1.0:
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
                self.stop_hand(self.hand_to_play, reward)
            self.stop_hand(self.hand_to_play, reward)

        elif action == 2:  # Double down
            hand = self.get_hand_to_play(self.hand_to_play)

            if not can_double_down(hand):
                reward = -20
                self.stop_hand(self.hand_to_play, reward)

            else:
                hand.append(draw_card(self.np_random))

                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.np_random))

                reward = cmp(score(hand), score(self.dealer))
                if is_bust(hand):
                    reward = -2
                    self.stop_hand(self.hand_to_play, reward)

                elif self.sab and is_natural(hand) and not is_natural(self.dealer):
                    # Player automatically wins. Rules consistent with S&B
                    reward = 2.0
                    self.stop_hand(self.hand_to_play, reward)

                elif not self.sab and self.natural and is_natural(hand) and reward == 1.0:
                    # Natural gives extra points, but doesn't autowin. Legacy implementation
                    reward = 3
                    self.stop_hand(self.hand_to_play, reward)
                
                self.stop_hand(self.hand_to_play, reward)
                

        elif action == 3:
            if not can_split(self.hand0, self.hand1):
                reward = -20
                self.stop_hand(self.hand_to_play, reward)
            else:
                self.hand0, self.hand1 = [self.hand0[0]], [self.hand0[1]]
                self.hand0.append(draw_card(self.np_random))
                self.hand1.append(draw_card(self.np_random))
                reward = 0

        self.set_hand_to_play(self.hand_to_play)
        return self._get_obs(), reward, self.is_done(), {"final_reward0": self.final_reward0, "final_reward1": self.final_reward1, "num_hand": 2 if self.has_split() else 1} if self.is_done() else {}

    def _get_obs(self):
        return (
            sum_hand(self.hand0),
            self.dealer[0],
            usable_ace(self.hand0),
            can_double_down(self.hand0),
            sum_hand(self.hand1),
            usable_ace(self.hand1),
            can_double_down(self.hand1),
            can_split(self.hand0, self.hand1),
            self.hand_to_play,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.dealer = draw_hand(self.np_random)
        self.hand0 = draw_hand(self.np_random)
        self.hand1 = []
        self.hand_to_play = 0
        self.hand0_stop = False
        self.hand1_stop = False
        self.final_reward0 = None
        self.final_reward1 = None
        if not return_info:
            return self._get_obs()
        else:
            return self._get_obs(), {}

    def render(self, mode="human"):
        player_sum, dealer_card_value, usable_ace = self._get_obs()
        screen_width, screen_height = 600, 500
        card_img_height = screen_height // 3
        card_img_width = int(card_img_height * 142 / 197)
        spacing = screen_height // 20

        bg_color = (7, 99, 36)
        white = (255, 255, 255)

        if not hasattr(self, "screen"):
            if mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                pygame.font.init()
                self.screen = pygame.Surface((screen_width, screen_height))

        self.screen.fill(bg_color)

        def get_image(path):
            cwd = os.path.dirname(__file__)
            image = pygame.image.load(os.path.join(cwd, path))
            return image

        def get_font(path, size):
            cwd = os.path.dirname(__file__)
            font = pygame.font.Font(os.path.join(cwd, path), size)
            return font

        small_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 15)
        dealer_text = small_font.render("Dealer: " + str(dealer_card_value), True, white)
        dealer_text_rect = self.screen.blit(dealer_text, (spacing, spacing))

        suits = ["C", "D", "H", "S"]
        dealer_card_suit = self.np_random.choice(suits)

        if dealer_card_value == 1:
            dealer_card_value_str = "A"
        elif dealer_card_value == 10:
            dealer_card_value_str = self.np_random.choice(["J", "Q", "K"])
        else:
            dealer_card_value_str = str(dealer_card_value)

        def scale_card_img(card_img):
            return pygame.transform.scale(card_img, (card_img_width, card_img_height))

        dealer_card_img = scale_card_img(
            get_image(os.path.join("img", dealer_card_suit + dealer_card_value_str + ".png"))
        )
        dealer_card_rect = self.screen.blit(
            dealer_card_img,
            (
                screen_width // 2 - card_img_width - spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        hidden_card_img = scale_card_img(get_image(os.path.join("img", "Card.png")))
        self.screen.blit(
            hidden_card_img,
            (
                screen_width // 2 + spacing // 2,
                dealer_text_rect.bottom + spacing,
            ),
        )

        player_text = small_font.render("Player", True, white)
        player_text_rect = self.screen.blit(player_text, (spacing, dealer_card_rect.bottom + 1.5 * spacing))

        large_font = get_font(os.path.join("font", "Minecraft.ttf"), screen_height // 6)
        player_sum_text = large_font.render(str(player_sum), True, white)
        player_sum_text_rect = self.screen.blit(
            player_sum_text,
            (
                screen_width // 2 - player_sum_text.get_width() // 2,
                player_text_rect.bottom + spacing,
            ),
        )

        if usable_ace:
            usable_ace_text = small_font.render("usable ace", True, white)
            self.screen.blit(
                usable_ace_text,
                (
                    screen_width // 2 - usable_ace_text.get_width() // 2,
                    player_sum_text_rect.bottom + spacing // 2,
                ),
            )
        if mode == "human":
            pygame.display.update()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))
