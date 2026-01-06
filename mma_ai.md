1/6/26, 9:08 AM MMA-ALnet

MMA-AI net Predictions About News

https://www.mma-ai.net/news 1/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

2025's Most Extreme Fighter Outliers: The
Statistical Freaks

January 2025

This breakdown comes out of the same pipeline | use for my UFC prediction models: a
big PostgreSQL feature store, opponent-adjusted stats, time decay, and a ton of
Bayesian smoothing.

I'm not claiming to be an official analyst or anything like that. This is just my best effort to point the matr
at the UFC and say:

"Okay, which fighters look genuinely weird for their weight class right now?"

If something seems off, or you've got ideas for better thresholds or features, feel free to roast or sugges
improvements in the comments. I'm iterating this all the time.

What Is Adjusted Performance (AdjPerf)?

Everything in here is driven by adjusted performance, or adjperf.
Ata high level, adjperf answers:

"How did this fighter perform compared to what their opponent normally allows, within their weight

class?"

Plain English version:
1. Every opponent has a history of what they usually allow:

© How many sig strikes per minute they eat
° How many takedowns they give up

© How often they get subbed, dropped, controlled, etc.

2. Ina given fight, we compare what our fighter did vs that opponent's "allowed" baseline.

3. We normalize that difference using a robust measure of spread (MAD, not vanilla standard
deviation).

4. We get a z-score inside that weight class:

© 0 =perfectly average for that matchup
© +#1to +2 = clearly above average

° +34 = extreme

© Scores are clipped to #7 so they don't blow up the model

2/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

High adjperf = you did something to that opponent that other fighters in your weight class usually
cannot do.

ADJUSTED PERFORMANCE (ADJPERF): THE SIMPLE VERSION

Al

oe THE RAW ACTION
(eg. Landing @ Punch)
OPPONENT: OPPONENT:
GOOD DEFENDER POOR DEFENDER
(Hard to Hit) (Easy to Hit)

KEY TAKEAWAY: Doing something difficult against a tough opponent is worth much more than doing
the same thing against an easy one. Adjperf measures the *difficulty* of the achievement.

Simplified overview of how adjperf works

How AdjPerf Actually Works (Nerd Corner)

If you like the guts, here's the rough shape of the math.

For a given stat (say, takedowns per minute), we compute:
adjperf = clip((observed - w_shrunk) / MAD_shrunk, -7, +7)

Where:

+ Observed = the fighter's stat in that fight (already on a canonical per-minute / rate scale).

* Opponent history (allowed):

© We look at what that opponent's past opponents have done against them.

© Compute an opponent-level mean and MAD of “allowed values, with optional time-decay so
recent fights matter more.

+ Weight-class prior:
© We have a weight-class mean and MAD for that stat (flyweight, bantamweight, etc.).
+ Bayesian shrinkage:
© We blend opponent history and weight-class prior:
wen / (n+ K)

where n is effective sample size (Kish-adjusted under decay).
© Small n > we lean more on the weight-class prior.
© Large n > we trust the opponent-specific history.

© That gives:

3/79


1/6/26, 9:08 AM MMA-ALnet

ushrunk = w_mean + opp + (1 - wmean) + pwe

MAD_shrunk = max(w_mad + MAD_opp + (1 ~ w.mad) + MAD_wc, MAD_floor)
* Clipping:
© Final z-scores are clipped to +7 in the feature store so extreme fights don't dominate training.
So adjperf is:

* Opponent-aware
+ Weight-class-aware
+ Time-aware (via decay)

* And robust (using MAD + shrinkage + clipping)

It's not just "he had a big fight once" — it's “given who he fought, in this weight class, how far off the
expected distribution was that performance?"

UNDERSTANDING ADJUSTED PERFORMANCE (ADJPERF): CONTEXT MATTERS

RAW PERFORMANCE ADJUSTED PERFORMANCE
(SURFACE LEVEL) (TRUE VALUE)
"ra sTmmas
sims srbec5 rites srt,
cries eviucs Gown al
2 rence
sngistwenr
Looks
IDENTICAL?
f FGHTERA —gePeNEE. OHTERB OPPONENT,
‘GnsttIno (GASELING
FIGHTERA OPPONENT OPPONGNTY FIGHTER
/ADJPERF SCORE: HIGH (+) 'ADJPERF SCORE: LOWER (-)
Z-SCORE (eg, +35) Z-SCORE (eg, 1.0)
without context, both seem equally good H Much better than expected Worse than expected
i against a tough opponent ‘against an easier opponent.
Highly impressive. Less impressive.

KEY TAKEAWAY: The Adjusted Score (Z-Score) measures performance relative
to what the opponent normally allows, revealing true skill beyond rav numbers.
Its about defying expectations, not just totals.

Detailed technical breakdown of the adjperf calculation pipeline

For this article, I'm looking at time-decayed adjperf so we're capturing current form, not ancient histor

How Fighters Made This List

To show up here, a fighter had to:
1, Be a clear adjperf outlier in their weight class

© Their most recent time-decayed adjpert for that stat is way out on the tail.

© We're talking "this would make a data scientist raise an eyebrow’ levels.
2. Have raw stats that tell the same story

© Attempts, land rates, per-minute numbers from actual fight logs line up with the adjperf
narrative.

Adjpert itself is already the primary signal — it's built to be stable and opponent-aware. The raw stats
you see below are there to make the story readable and to give you something concrete to sanity-check

https://www.mma-ai.net/news 4/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

against (and yell at me about if you disagree).

The 13 Most Extreme Statistical Outliers of 2025

1. Valter Walker — Heavyweight Submi

Stat: Submission attempts per minute

Most Recent AdjPerf: +53.60 (pre-clipping)
Weight Class: Heavyweight

UFC Fights: 5
Most Recent Fig!

: 2025-10-25

Heavyweight is supposed to be overhands and brain cells leaving the building. Walker is out here playing
“lightweight jiu-jitsu specialist" instead.

Raw rate: 0.561 submission attempts per minute

+ Attempts by fight: 1, 1, 1, 1,0
+ Fight times: 1.4, 0.9, 1.3, 49, 15.0 minutes

Against heavyweights who usually don't even give up many sub attempts, this is wild. Adjperf is basicall
saying:

"Relative to what heavyweights normally allow, this guy is constantly hunting submissions in a divisio
that does not do that."

He's a genuine grappling outlier at heavyweight.

2. Anshul Jubli- Lightweight Touch-of-Death Candidate

Stat: KO efficiency (KOs per sig strike landed)
Most Recent AdjPerf: +31.73

Weight Class: Lightweight

UFC Fights: 3
Most Recent Fig!

: 2025-02-08

Most lightweights need accumulation. Jubli just needs the right connection.

* 1KO from 141 sig strikes total (0.7% overall)
+ Inthe KO fight: 1 KO from 38 sig strikes (~2.6% in that bout)

Adjperf adjusts for who he knocked out and how hard they are to finish. The model is essentially
screaming "this wasn't just some glass-chin mercy-stoppage."

Small sample size, yes. But early data says: when he lands clean, it ends badly for people.

3. Merab Dvalishvili - Takedown Volume Barbarian

Stat: Takedown attempts per minute
Most Recent AdjPerf: +6.53

5/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
Weight Class: Bantamweight
UFC Fights: 17
Most Recent Fight: 2025-12-06

Most fighters shoot to mix things up. Merab shoots as a lifestyle.

* 117 takedown attempts over his UFC run
+ 1.085 attempts per minute

+ Recent 25-minute fights: 29, 37, 30, 15 attempts

Adjperf compares that volume to what his opponents usually allow. Elite defensive wrestlers who normal
see ~0.3 attempts/min are getting bombarded at over 1 attempt/min.

Even if you're sitting on elite takedown defense, math starts to cave in when someone chains 30+
attempts on you. You defend, you get up, you defend again, forever.

4. Jailton Almeida - Heavyweight Takedown Factory

Stat: Takedowns landed per minute
Most Recent AdjPerf: +11.37
Weight Class: Heavyweight

UFC Fights: 10

Most Recent Fight: 2025-10-25

First he gets you down. Then you stay there. The first part is this section.

+ 34 takedowns total
+ 0.514 takedowns per minute

+ Examples:

© 7 takedowns in a 15-minute fight

© 6 ina 25-minute fight
Against heavyweights who normally do not get taken down much, that's a big departure from the baselit

Adjperf is calling out that he's consistently converting takedowns at a rate his opponents do not usually
allow — and that's the on-ramp to his entire submission game.

5. Hamdy Abdelwahab - "If | Shoot, You

Stat: Takedown accuracy
Most Recent AdjPerf: +26.68
Weight Class: Heavyweight
UFC Fights: 4
Most Recent Fig!

: 2025-10-25

Hamdy's story isn't volume; it's precision.

+ 18 takedowns on 20 attempts (65.0% accuracy)
+ Most recent fight: 9/14 (64.3%)

Heavyweights are usually pretty good at just being big and hard to drag down. Adjusted for opponent
TDD, +26.68 is the model saying "these guys don't normally get hit with this many clean entries."

6/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

It's still an early sample, but the technique and efficiency are tracking as outliers.

6. Kyoji Horiguchi — Flyweight Knockdown Glitch

Stat: Knockdowns per minute
Most Recent AdjPerf (pre-clip): +945.00
Weight Class: Flyweight

UFC Fights: 9

Most Recent Fight: 2025-11-22

This is the single craziest value in the whole system before clipping. Here's why it explodes:

+ Knockdowns at 125 are extremely rare.
+ That makes the weight-class MAD for KD/min tiny.

* When someone genuinely overperforms there, the un-clipped z-score goes to the moon.

In practice:

* 6 knockdowns in 9 UFC fights
* Career KD rate: 0.053 per minute
+ 2025 fight: 2 knockdowns in 12.3 minutes (0.163 KD/min)

In the live feature store, this gets clipped at +7 for sanity, but the underlying signal is still: for flyweight,
this guy's knockdown production is absurd.

And he's not just raw power — it's timing, entries, and shot selection built on years of striking experience

7. Joshua Van - Endless Output at Flyweight

Stat: Significant strikes landed per minute
Most Recent AdjPerf: +5.30

Weight Class: Flyweight

UFC Fights: 10

Most Recent Fight: 2025-12-06

Van fights like the slider for "pace" is stuck all the way to the right.

* 1,099 sig strikes total

* 8.354 sig strikes per minute

+ Recent fights: 204 strikes (15 min), 125 strikes (14 min), 165 strikes (15 min)
Adjperf compares that to what his opponents usually allow and flags him as a clear outlier. These are fas
defensively sharp flyweights, and he's still landing at a pace most guys can't match for a round, never

mind three.

His win condition is simple: survive his pace or break.

8.Ciryl Gane - Overall Accuracy Cheat Code

Stat: Overall strike accuracy
Most Recent AdjPerf: +2.93

179


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
Weight Class: Heavyweight
UFC Fights: 13
Most Recent Fight: 2025-10-25

Gane is what happens when a heavyweight actually cares about clean shot selection.

+ 929 of 1,507 total strikes landed (61.6% accuracy)

* Last three fights: 75.0%, 70.5%, 69.9% total accuracy

Adjperf is factoring in the fact that his opponents typically don't get lit up at these percentages. For
heavyweight, where a lot of guys swing big and miss a lot, this level of sustained efficiency is unusual.

Volume isn't crazy; the quality of each attempt is.

9. Ciryl Gane -

stance Sniper

Stat: Distance strike accuracy
Most Recent AdjPerf: +4.04
Weight Class: Heavyweight
UFC Fights: 13
Most Recent Fig!

: 2025-10-25
If #8 is the macro view, this is his true specialty: long-range work.

* 837 of 1,394 distance strikes landed (60.0% distance accuracy)
+ Recent fights at distance: 75.0%, 70.2%, 69.1%

Most heavyweights are happy to crack 40-45% at distance against top guys. Gane is living in the 60-7
range.

Adjperf is saying: even after adjusting for how little these opponents normally let others hit them at rang
Ganeis still an outlier.

He appears twice on this list because his game is systematically efficient, not just one big performance
skating the numbers.

10. Payton Talbott - Headshot Specialist at Bantamweight

Stat: Head strike accuracy
Most Recent AdjPerf: +2.91
Weight Class: Bantamweight
UFC Fights: 6

Most Recent Fight: 2025-12-06

Talbott doesn't just land; he finds the head consistently.

+ 262 of 498 head strikes landed (52.6% head accuracy)
* Most recent fight: 89 of 167 to the head (53.3%)

Bantamweight is full of guys with great head movement, footwork, and defense. Maintaining 50%+ head
accuracy fight after fight against that kind of opposition is rare.

Adjperf incorporates opponent defensive quality and still puts him clearly above the curve. High volume
plus high precision to the most important scoring target is exactly what judges (and knockouts) like.

8/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

11. Jean Matsumoto — Leg Kick Merchant

Stat: Leg strike share (% of total strikes to the legs)
Most Recent AdjPerf: +3.17

Weight Class: Bantamweight

UFC Fights: 4
Most Recent Fig!

: 2025-08-09
Matsumoto doesn't just "use" leg kicks; he builds entire days around them.
Per fight:

+ Leg kicks: 19, 25, 21,8
* Total sig strikes: 95, 77, 89, 19
+ Leg kick share: 20.0%, 32.5%, 23.6%, 42.1%

Adjperf compares his leg-kick focus to what opponents usually face and flags him as a specialization
outlier. He's still going hard to the legs even against guys who historically defend or avoid them well.

This is classic Dutch kickboxing logic applied in MMA: attack the base until the opponent can't move or
plant properly late in the fight.

12. Kyoji Horiguchi — Top-Game Controller at Flyweight

Stat: Ground strike percentage (% of sig strikes from ground positions)
Most Recent AdjPerf: +5.79

Weight Class: Flyweight

UFC Fights: 9
Most Recent Fig!

1 2025-11-22
Same guy as the knockdown glitch, now on the grappling axis.
+ 2025 fight: 24 of 49 sig strikes from the ground (49.0%)

Adjperf is adjusting for the fact that his opponents typically keep fights standing against most other
people. Horiguchi is dragging them into long stretches of top control and doing meaningful work from
there.

At flyweight, where scrambling and standups are usually constant, being able to actually hold people
down and build offense is a skill that really stands out in the data.

He's on this list twice because he's extreme in two totally different ways: dropping people and riding
them.

13. Jailton Almeida — Control Time Tyrant

Stat: Control time per minute
Most Recent AdjPerf: +10.12
Weight Class: Heavyweight
UFC Fights: 10

Most Recent Fight: 2025-10-25

We covered his takedowns already. Control time is the second half of the nightmare.

9/79


1/6/26, 9:08 AM

Articles

2025's Most Extreme Fighter
Outliers: The Statistical Freaks

2024 vs 2025: The Year UFC
Betting Markets Got More
Precise, But Less Accurate

How to Install AutoGluon 1.4 if
You're Using an Nvidia 5090
Card

Data Drift, Generalization, and
the Quest for a Bulletproof UFC
Model

Calibrating UFC Fight Predictions

Machine Learning for Sports
Prediction: Should You Balance
the Winrate of Competitor 1 vs
Competitor 2?

Al Predictions and Analysis for
UFC: Emmett vs Murphy

New Video: Understanding
MMA-Al Predictions

Demystifying the MMA-AI
Prediction Algorithm: How We
Predict UFC Fights

Unit Testing Not Done Yet
v5.2 Model Release
Updates to Betting Strategy

Welcome to the new MMA-Al.net

https://www.mma-ai.net/news

MMA-ALnet

+ 3,759 total seconds of control
+ 49.775 seconds of control per minute (out of 60)

+ Recent examples:

© 647 seconds of control in a 15-minute fight (~10.8 minutes)

© 1,270 seconds ina 25-minute fight (~21.2 minutes)

Adjperf measures this against how much control his opponents usually give up. For most heavyweights,
being stuck under someone that long is unusual; for Almeida, it's just the pattern.

He shows up twice because in the data he's basically:

1. Getting you down at an above-normal rate, and

2. Keeping you there at an above-normal rate.

From a modeling perspective, that's a very clear and very stable identity.

Quick Glossary

Z-score (a)
How many robust standard deviations (via MAD) above or below the weight-class average a performanc
is:

+ +3.0 and up: extreme outlier

+ +2.0o to +3.0a: elite
+ +#1.5¢ to 42.00: very strong

+ +#1.0c+: clearly above average

Adjusted Performance (adjperf)
Opponent-adjusted, weight-class-adjusted, time-decayed z-score:

* Centers on what that opponent typically allows (with shrinkage toward weight-class norms).
* Uses MAD instead of standard deviation for robustness.
* Clipped to +7 to keep the model stable.

Time-decayed average (dec_avg)

A rolling average where recent fights get more weight. This gives you "who this fighter is now" instead o

letting old fights dominate.

Weight-class comparison
All adjperf calculations are done inside the fighter's own weight class:

+ Flyweights are compared to other flyweights.
+ Heavyweights to heavyweights.

+ No cross-weight-class nonsense.

Validation Philosophy

For the model, adjperf is the primary signal. It's designed to already reflect:

10/79


1/6/26, 9:08 AM MMA-ALnet

Opponent quality

Weight-class context

Sample size and volatility

Recency
For this article, | layered on a simple sanity check:

1. Adjperf says "this is an outlier.”

2. check that the raw fight stats (attempts, land rates, per-minute values) tell a story that matche
what the metric is flagging.

If adjperf had someone barely above average and the write-up would be pure cope, they didn't make the
list. If adjperf had someone pinned near the ceiling but the fight log clearly didn't fit the narrative, I left
them out as well

This is my best attempt to blend a pretty serious statistical pipeline with writeups that still conne

to what you see on tape. If you think | missed someone, or you've got ideas on better stats to adjperf-i
next, drop it in the comments — I'm always tuning this.

https://www.mma-ai.net/news 1/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

2024 vs 2025: The Year UFC Betting Markets
Got More Precise, But Less Accurate

January 2025

| gave Al access to my odds database and told it to do analysis on the data. This is the
result. | double checked most of the results but bear in mind there might still be minor
discrepancies and also note that my odds data is not 100% comprehensive, | think |
have about 70%(?) coverage of all the fights in 2024/2025.

TL;DR: The Paradox

2025 was more precise, but less accurate. 2024 saw Vegas accuracy at 71% and 2025 saw
Vegas accuracy at 64%, a startling decline. Betting markets became more efficient (less line
movement, fewer extreme swings), suggesting sharper pricing. However, Vegas's predictions
actually got worse—favorites won less often, and the Brier score (a measure of prediction
accuracy) increased, meaning less accurate forecasts.

The Core Paradox

When we analyze betting markets, we typically expect efficiency and accuracy to go hand-in-hand. Mor
efficient markets should mean better predictions. But 2025 broke that assumption.

The Paradox: Markets More Efficient, But Vegas Less Accurate

42> - 0220

Less Accurate | ozs

Loz10

More Efficient

0205

=
3
2
a
3
2

e200

+ 0.198
36+

0.190
2024 2025
Year

Figure 1: Markets became more efficient (median movement decreased), but Vegas accuracy declined (Brie
score increased)

12/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Key Finding: Median absolute line movement decreased from 4.05% to 3.71% (markets more

efficient), while the Brier score increased from 0.1980 to 0.2136 (Vegas less accurate).

Where Did Vegas Go Wrong?

To understand why Vegas became less accurate despite more efficient markets, we need to look at
calibration—how welll Vegas's implied probabilities matched actual outcomes.

Vegas Calibration: Expected vs Actual Win Rates by Probability Bucket

Perfect Calsration

ot ee Ls

% 00.90%

8.20%

70-50%

a

Actual Win Rate (%)
8

oo 70 80 0 1
Expected Win Rate (%)

Figure 2: Expected vs Actual Win Rates by Probability Bucket. Points on the diagonal line = perfect calibratic

The calibration plot reveals a critical issue: Vegas significantly underestimated favorites in 2025,
particularly in the 60-70% probability bucket. In 2024, Vegas was well-calibrated for this range (64.9%
actual vs 65.1% expected). But in 2025, favorites in this bucket won only 58.0% of the time when Vegas
expected 65.5%—a 7.5 percentage point miss.

The Problem: Vegas overestimated the likelihood of favorites winning in 2025, especially those
priced between 60-70% implied probability. This suggests Vegas may have been too confident in
favorites or underestimated the competitiveness of matchups.

Sharp Money Influence Declining

One explanation for more efficient markets is that sharp bettors (professional bettors with sophisticated
models) had less influence in 2025. When sharp money moves lines significantly, it creates volatility. Les
sharp influence means tighter, more stable lines.

13/79


1/6/26, 9:08 AM MMA-ALnet
‘Sharp Money Influence Declining: 2024 vs 2025

7 i
.
Be
i
a.
re
a
ae
mo
Ea een

Metric

Figure 3: Sharp money influence declined across all metrics in 2025

* Average Movement: Decreased from 5.34% to 4.80% (-10.2%)
+ Steam Bets: Decreased from 13.8% to 10.1% (-37 percentage points)

* Extreme Movements (>20%): Decreased from 1.4% to 0.9% (-50%)

"Steam bets" are rapid, coordinated line movements that typically indicate sharp money entering the
market. The decline in steam bets suggests either:

1. Sharps found fewer value opportunities (markets already efficient)
2. Sharps were less active in 2025

3. Public bettors got smarter, reducing the need for sharp correction

Public Getting Smarter: Favorites Moving Against
Public Money

One of the most interesting trends in 2025 was the increase in favorites whose odds worsened (got
longer) despite being the public favorite. This is a key indicator of sharp money betting against public
sentiment.

https://www.mma-ai.net/news 14/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
Publ

Getting Smarter: More Favorites Moving Against Public
100

Favorites Moving WITH Public
mm Favortes Moving AGAINST Public

70.1%

Percentage of Favorites

62.7%

Year

Figure 4: Percentage of favorites moving with vs against public money

Understanding "Favorites Moving With vs Against Public"

Favorites Moving WITH the Public: When a fighter opens as the favorite and their odds get better
(lower odds, higher implied probability) as the fight approaches. This happens when public money
floods in on the favorite, driving the line in their favor. Example: A fighter opens at +150 (40%
implied) and closes at +120 (45% implied) because everyone bets on them.

Favorites Moving AGAINST the Public: When a fighter opens as the favorite but their odds get
worse (higher odds, lower implied probability) as the fight approaches. This is a classic sign of sharp
money betting against the public favorite. Example: A fighter opens at -200 (67% implied) but
closes at -150 (60% implied) because sharps bet the underdog, moving the line against public
sentiment.

Why This Matters: When favorites move against public money, it often means sharp bettors see
value in the underdog. The increase from 29.9% to 37.3% suggests either sharps found more value
betting against favorites, or public bettors were less informed, creating more opportunities for sharp
correction.

In 2024, 29.9% of favorites moved against public money. In 2025, this increased to 37.3 %—a +7.4
percentage point increase. This suggests sharp bettors found more value betting against favorites, or th
public bettors were less accurate in their initial assessments.

Insight: The increase in favorites moving against public money, combined with declining sharp
influence metrics, creates an interesting contradiction. It's possible that sharps were more selective
—betting bigger on fewer opportunities, creating larger movements when they did bet, but betting
less frequently overall.

Temporal Patterns: When Steam Happens

Not all months are created equal when it comes to betting activity. February and October 2025 saw the
highest steam bet rates, suggesting these months had the most sharp betting activity or the most

15/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

mispriced lines.

‘Monthly Steam Bet Patterns: 2025
ewes swan ae

Fr Pak San a +

er 162%

‘Steam Rate (4)

Fe Mr ne uy ry
Month

Figure 5: Monthly steam bet rates in 2025, with fight counts shown in background

February 2025: 25.0% steam rate—the highest of the year. This suggests significant sharp activity
major line corrections during this period.

October 2025: 16.2% steam rate—the second highest. This could indicate high-profile fights with
significant betting interest or mispriced lines.

July 2025: 0.0% steam rate—the lowest, with only 18 fights. This suggests either very efficient initial
pricing or limited sharp interest.

The Biggest Line Movements of 2025

Some fights saw dramatic line movements, indicating significant sharp money or late-breaking
information. Here are the most extreme cases:

‘Top 10 Biggest Line Movements: 2025

i Sh 4648 @
Noon Toe nl
omer tates 239% @
Sout Comer 29% @—$$
pie — 946%
‘ince Maes SSS
os in ts
sess Dane irr

otyGbsen rs

Implied Probability Movement ()

Figure 6: Top 10 biggest implied probability movements in 2025

16/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Biggest Positive Movement: Cody Gibson saw his implied probability increase by +25.05% (from

40.8% to 65.9%) against Aorigileng. This massive move suggests sharp money heavily favored Gibson,
late-breaking information changed the assessment.

Biggest Negative Movement: Aorigileng (in the same fight) saw his implied probability decrease by

-24.65% (from 63.0% to 38.3%). This is the flip side of Gibson's movement—when one fighter's odk
improve dramatically, their opponent's must worsen.

Opening vs Closing: Market Learning

One way to measure market efficiency is to compare opening odds (the initial line) to closing odds (the
final line). If markets are learning and incorporating information, closing odds should be more accurate
than opening odds.

Opening vs Closing Odds Accuracy: Market Learning
0220
Opening Ove
fm Closing Ons

0.0012

o2is 2145

0210

Better)

40.0065,

0.208

Brier Score (Lower

0.198,

Year

Figure 7: Brier scores for opening vs closing odds. Lower Brier score = better accuracy

In both years, closing odds were more accurate than opening odds, confirming that markets do learn an
incorporate information. However, the improvement was much smaller in 2025:

* 2024: Opening Brier 0.2045 Closing Brier 0.1980 (+0.0065 improvement)
* 2025: Opening Brier 0.2148 > Closing Brier 0.2136 (+0.0012 improvement)

This suggests that in 2025, opening lines were already closer to optimal, leaving less room for
improvement. However, the overall accuracy was still worse than 2024, indicating a systematic calibratic
issue rather than just initial mispricing.

Fighter-Specific Storylines

Some fighters consistently saw significant line movement, suggesting the market was frequently
mispricing their fights or that sharp bettors consistently found value betting on (or against) them.

17/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Top 10 Fighters by Average Line Movement: 2025 (Min 3 Fights)

‘Nexander Herande2. 5.36%

Reon Barcelo 552%

Melquizael Costa 5.58%

men Shahbazyan 5.06%

Kevin Holand 637%

Reiner De Rider 6.62%

Cody Brundage

3

Brendson Ribeiro 7.36%

Hamdy Abdetwahab 7.88%

Jose Delgado 10.50%

4 6 8 10
Average Absolute Movement (%)

Figure 8: Top 10 fighters by average absolute line movement in 2025 (minimum 3 fights)

Jose Delgado led the pack with an average movement of 10.58% across 3 fights, with odds improvi

in 2 of those fights. This suggests the market consistently undervalued Delgado initially, and sharp mon
corrected the line.

Distribution of Movements: Less Extreme Tails

The overall distribution of line movements shifted in 2025, with fewer extreme movements and a tighter
distribution around the median.

Distribution of Line Movements: Less Extreme Tails in 2025

40

8

Absolute IP Movement (%)

2008 2025

Figure 9: Distribution of absolute line movements. Wider violins = more variation

18/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
The violin plot shows that 2025 had a tighter distribution with less extreme tail movements. The median

decreased from 4.05% to 3.71%, and the standard deviation decreased fron 4.90% to 4.12%

This confirms that markets became more stable and efficient, with fewer dramatic corrections.

Movement vs Accuracy: Individual Fight Analysis

Does more line movement correlate with better or worse accuracy? The scatter plot shows the
relationship between absolute movement and prediction error for individual fights.

Line Movement vs Prediction Accuracy: Individual Fights

‘Absolute IP Movement (%)

Figure 10: Relationship between line movement and prediction accuracy for individual fights

While there's no strong correlation visible, the plot shows that both years had fights with high movemen
and low error (sharp corrections that improved accuracy) as well as high movement and high error
(corrections that made things worse). This suggests that movement alone doesn't guarantee accuracy-
depends on whether the movement is correcting a mispricing or overreacting to information.

Most Suspicious Fights: Statistical Anomalies

While analyzing the data, we identified fights with patterns that raise statistical eyebrows—combinations
of factors that, while not proof of anything nefarious, are worth noting. We created a composite “suspici
score" that combines:

+ Prediction Error (30%): How wrong Vegas was about the outcome

+ Late Movement (25%): Dramatic odds changes in the final 7 days before the fight
+ Heavy Favorite Loss (20%): Favorites with >70% implied probability that lost

+ Extreme Favorite Loss (15%): Favorites with >80% implied probability that lost

+ Underdog Late Win (10%): Underdogs who won with >5% late movement

Important Disclaimer

19/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Suspicious patterns don't necessarily indicate rig
explained by:

g. These statistical anomalies could also be

+ Sharp bettors with superior analysis or inside information
+ Late-breaking news (injuries, weight issues, etc.)
+ Legitimate upsets in a sport known for unpredictability

+ Market inefficiencies that sharp money correctly identified

The suspicion score simply identifies fights where multiple unusual factors converged—worth
investigating, but not proof of wrongdoing.

us Fights: 2024

Most Suspicious: Kyler Phillips vs Rob Font (October 19, 2024)

Suspicion Score: 0.8096 (highest of the year)

Context: Phillips was an 82.4% favorite but lost

Late Movement: 12.16% movement in final 7 days

+ Prediction Error: 0.6784 (Vegas was very wrong)

Pattern: Extreme favorite loss with significant late movement
Other notable suspicious fights in 2024:

+ Javid Basharat vs Aiemann Zahabi (Score: 0.7596) - 85.8% favorite lost
* Mateusz Rebecki vs Diego Ferreira (Score: 0.7536) - 81.6% favorite lost with 8.80% late moveme

* Kennedy Nzechukwu vs Ovince Saint Preux (Score: 0.6974) - 83.7% favorite lost

2024 Summary: 13 fights had suspicion scores >0.5, including 6 extreme favorite losses (>80%
IP). Mean suspicion score: 0.1511.

Top Suspicious Fights: 2025

Most Suspicious: Rinya Nakamura vs Muin Gafurov (January 18, 2025)

Suspicion Score: 0.7231 (highest of the year)

Context: Nakamura was an 84.1% favorite but lost

Late Movement: 7.14% movement in final 7 days

Prediction Error: 0.7072 (Vegas was very wrong)

Pattern: Extreme favorite loss

Other notable suspicious fights in 2025:

+ Josias Musasa vs Carlos Vera (Score: 0.6907) - 85.2% favorite lost
* Oumar Sy vs Alonzo Menifield (Score: 0.6875) - 83.8% favorite lost

+ Payton Talbott vs Raoni Barcelos (Score: 0.6523) - 88.4% favorite lost (highest IP favorite to lose

2025 Summary: 9 fights had suspicion scores >0.5, including 5 extreme favorite losses (>80% IP).
Mean suspicion score: 0.1534—slightly higher than 2024, suggesting similar levels of statistical

20/79


1/6/26, 9:08 AM MMA-ALnet

anomalies.

Patterns in Suspicious Fights

Analyzing the suspicious fights reveals some patterns:

+ Heavy favorites losing: Most suspicious fights involve favorites with >80% implied probability losir
—these are inherently rare events that Vegas heavily favored.

+ Late movement correlation: Many suspicious fights had significant late movement (>5%),
suggesting information entered the market close to fight time.

+ Underdog wins: Several suspicious fights featured underdogs winning with large late movement,
potentially indicating sharp money betting against public sentiment.

Notable: Payton Talbott vs Raoni Barcelos stands out—Talbott was an 88.4% favorite (one of the highes

IP favorites in the dataset) but lost. This represents a massive prediction error (0.7823) even though the
was minimal late movement (0.16%), suggesting the initial line may have been fundamentally mispriced.

Conclusion: The Precision-Accuracy Paradox

What We Learned

2025 was more precise, but less accurate. Betting markets became more efficient—lines moved
less, extreme swings decreased, and markets appeared more stable. This suggests sharper initial
pricing and less need for correction.

However, Vegas's predictions became less accurate. Favorites won less often, and the Brier score
increased. The calibration plot revealed that Vegas significantly underestimated favorites in the
60-70% probability range, suggesting a systematic overconfidence in favorites.

Why the Paradox? Several factors likely contributed:

+ More efficient markets mean less room for sharp correction, but also less information
incorporation. If opening lines are already tight, there's less opportunity for markets to learn.

+ Sharp money declining could mean fewer corrections to public mispricing, allowing
systematic biases to persist.

* Public getting smarter (or less active) might mean less public-driven mispricing, but also
less information flow that sharps typically exploit.

* Vegas calibration issues suggest the sportsbooks may have been systematically
overconfident in favorites, possibly due to model changes or different risk management
approaches.

The Bottom Line: Efficiency doesn't always equal accuracy. More stable, precise markets can still
produce less accurate predictions if there are systematic biases in the underlying models. For
bettors, this paradox suggests that even in more efficient markets, there may be opportunities—
especially if you can identify and exploit systematic calibration errors.

Analysis based on 802 fights in 2024 and 664 fights in 2025, using vigless implied probabilities from closing oda
Data sourced from features.odds table in mma-ai database.

https://www.mma-ai.net/news 21/79


1/6/26, 9:08 AM MMA-ALnet

https://www.mma-ai.net/news 22/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

How to Install AutoGluon 1.4 if You're Using a
Nvidia 5090 Card

November 22, 2025

If you're lucky enough to have an NVIDIA RTX 5090 (or any Blackwell architecture GPU), you've probabl
discovered that getting AutoGluon 1.4.0 to work with it isn't exactly straightforward. The problem?
AutoGluon requires older PyTorch versions, but the RTX 5090 needs PyTorch Nightly with CUDA 13
support. This guide will walk you through a "Frankenstein" installation that makes these incompatible
pieces work together.

Prerequisites
Before we dive in, make sure you have:
+ Python 3.10-3.12 (Python 3.12.4 recommended)

© Note: Python 3.10-3.12 required for PyTorch Nightly (RTX 5090 setup)
© AutoGluon 1.4.0 requires Python >=3.9,<3.13

© If using uv (recommended), Python will be installed automatically - no need to install it separat:

* PostgreSQL database
+ uv package manager (required for RTX 5090 setup)
+ NVIDIA GPU (RTX 5090 / Blackwell supported with special setup)

* Extreme Performance Setup (RTX 5090 / Blackwell)

A CRITICAL NOTE: This setup uses a "Frankenstein" environment to force AutoGluon (which normally
requires older PyTorch) to run on the RTX 5090 (which requires PyTorch Nightly/CUDA 13).

DoNOT run uv sync after completing these steps. Doing so will revert your packages and break
the environment.

Step 1: Configure Base Dependencies

Ensure your pyproject.toml includes torch (configured for Nightly) but EXCLUDES autogluon .

Install the base environment (Layer 1): This installs Python 3.12, PyTorch Nightly (CUDA 13), and standar
tools (pandas, numpy, etc.).

uv syne

Step 2: The "Sideload" Installation

We must manually force-install AutoGluon without dependencies to bypass the PyTorch version check,
then manually install the missing libraries.

Force Install AutoGluon:

uv pip install “autogluon[tabarena]==1.4.0" --no-deps

23/79


1/6/26, 9:08 AM MMA-ALnet

Install Required Sub-Dependencies:

These are the libraries AutoGluon needs to run that we skipped in step 2.

uv pip install autogluon-common==1.4.0 autogluon-core==1.4.0 autogluon-features==1.4.0 autogl
Install Manual Fixes:

These libraries are required for the prediction pipeline and specific model architectures.

uv pip install einx loguru networkx psutil defusedxml scipy

Step 3: Verification

Run this command to confirm your environment sees the RTX 5090:

uv run python -c "import torch; print(f'CUDA Available: {torch.cuda. is_available()}'); print(

Success Output: CUDA Available: True | Device: NVIDIA GeForce RTX 5090 | Version:
++ dev2025...+cu130

Standard Installation (Non-5090 GPUs)

If you are NOT using an RTX 5090, you can use the standard installation method.
GPU Setup (RTX 30/40 Series):

# After base installation, install PyTorch with CUDA
uv pip install torch torchvision torchaudio --index-url https: //download. pytorch.org/whl/cul21

Check PyTorch installation guide for the appropriate CUDA version for your GPU.

Installation
Option 1: Using uv (Recommended)

Note: You don't need Python pre-installed! uv can install and manage Python for you.

1. Install uv (if not already installed):

On Windows (PowerShell):

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.psi | iex"

On macOS/Linux:

curl -LsSf https://astral.sh/uv/install.sh | sh
2. Install Python 3.12.4 using uv (or any Python 3.9-3.12):

uv python install 3.12.4

Or install the latest Python 3.12:

https://www.mma-ai.net/news 24/79


1/6/26, 9:08 AM MMA-ALnet

uv python install 3.12

Note: AutoGluon 1.4.0 requires Python >=3.9,<3.13

3. Clone and navigate to the project:

git clone <repository-url>
cd mma-ai-db

4. Install dependencies (uv will automatically create a virtual environment):

uv syne

This command will:

© Create a virtual environment automatically
© Install all dependencies from pyproject.tomt

© Use the Python version managed by uv

5. Verify installation (optional):

# Check AutoGluon installation
uv run python -c "from autogluon.tabular import TabularPredictor; print('AutoGluon ins

# Check GPU support (if applicable)
uv run python -c “import torch; print('CUDA available:', torch.cuda.is_available())"

6. Run the application:

uv run python main.py

Or run tests:

uv run pytest

Why This Works

The RTX 5090 uses NVIDIA's new Blackwell architecture, which requires CUDA 13 and the latest PyTore
Nightly builds. However, AutoGluon 1.4.0 was built and tested against older, stable PyTorch versions. By
using --no-deps during AutoGluon installation, we bypass its dependency checks and manually install
compatible versions of its sub-packages. This creates a working environment where PyTorch Nightly cat
communicate with your GPU while AutoGluon runs on top of it.

It's not elegant, but it works. And when you have a $2,000 GPU sitting in your machine, sometimes you
need to get creative to make things work.

https://www.mma-ai.net/news 25/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

v7 Model Release

September 3, 2025

Plain-English Summary

1. Adjusted Performance got smarter about opponents. | no longer judge a fighter against a
generic average or the opponent's last fight. | build what that opponent typically allows and bler
it with the division norm. If the opponent is historically stingy, landing on them counts more
finally separated counts from binaries. Counts (strikes landed/attempted, takedowns,

NR

reversals, etc.) are stabilized as rates per minute. Binary outcomes (KO, win, decision, sub
landed) and control share get probability-style smoothing. Different problems, different tools.

w@

The pipeline order prevents "peeking." | smooth first, keep a temporary +_raw , compute
derived stats on the smoothed values, then drop the raw columns. Opponent features and
weight-class priors are in place before adjusted performance is computed.

B

Small samples don't hijack scores. When the opponent has little history, | lean more on the
weight class. As real data accumulates, it naturally takes over. Extreme one-off nights are clippe
so they don't dominate training.

Deep Technical Analysis (How to Reproduce It)

0) Pipeline Order (high level)

Create base fight stats and copy to a derived table (so the rest of the pipeline has a stable
surface)
Beta-Binomial smoothing for binary families (KO, win, decision, sub landed, control share). Ru

NR

this first so attempts like sub_att are still raw.

w@

Poisson-Gamma smoothing for count families (e.g., sig_str_land, td_land, sub_att, kd,

rev )

B

Rename: temporarily keep originals as +_raw , replace originals with smoothed values, compute
totalsjaccuracy/defense/ratios/per, then delete +_raw
Create feature-specific tables; compute "per" features; build opponent features (what others di

a

against this fighter)

2

Compute weight-class means, MAD, and a per-stat minimum MAD floor.
Adjusted Performance (non-decayed then decayed) on the feature-specific tables.

“N

1) Adjusted Performance

Old behavior (why it was wrong)

* Baseline came from the opponent's /ast fight row: (A.stat - B.prev_opp_avg) / B.prev_opp_ma
One quirky row could swing the score.

* First-fight opponents fell straight to a weight-class table. No reliability-based blending

+ Denominator was a single-row MAD with a hard floor. That prevents division explosions but
doesn't capture uncertainty.

New behavior (what to implement)

1. Opponent history per feature table. For a target column c (e.g., head_acc ), gather all past
fights where other fighters faced the current opponent. Compute:

© opp_mean_pers(c) : mean of what others achieved against this opponent. If time-decay

on, weight each row w = exp(-A-years) and use weighted mean.

26/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

© opp_mad_pers(c) : MAD via a two-step procedure: median of val , then median of |va
~ median| «
© n:sample size. With decay, use Kish effective n: n = (Zw)? / Sw.

-- skeleton of the per-column CTEs
rows_c (fight_id, fighter_id, val, w)

med_c (median per fight_id, fighter_id)

rows_with_m (join rows_¢ to med_c)

stats_c (mean val: SUM(w*val)/SUM(w) if decay, else AVG(val))
mad_c (median |val ~ median|)

2. Weight-class priors. Precompute for every feature table and column:

© c_we_mean, c_we_mad, and c_mad_floor .
-- each table: <table>_wc_mean, <table>_wc_mad, <table>_minimum_mad

3. Reliability-weighted shrinkage.

wmean =n / (n + K_mean)
wmad =n / (n + K_mad)

mu = w_mean * opp_mean_pers + (1 - w_mean) * wc_mean
sigma = max( w_mad * opp_mad_pers + (1 ~ w.mad) * wc_mad, mad_floor )

adjperf = clip( (observed - mu) / sigma , -7, 7 )

© Defaults used: Kean = 4.0, K_mad = 4.0 (tune per family if you want; often K_mad >
K_mean is safer).

© Observed is the already-smoothed feature value from the feature-specific table.

Data hygiene you must enforce

* Only use fights strictly before the current event; tie-break same-day events with (event_id,
fight_id) .

+ Ifa weight class lacks priors, fall back to a global row (don't coalesce to zero).

+ Recommended hardening: compute a per-column effective n and set the weight to zero if that
column has no history (instead of coalescing personal stats to 0).

2) Poisson-Gamma Smoothing (Counts)

Scope

Columns ending with _land or _att ,plus kd and rev (and their _rdi forms). Exclude static fields
and binary/duration families.

Exposure definitions

* _rd1 columns: t = min(time_sec_rd1, 300) / 60.0
* Allothers: t = time_sec / 60.0

Priors as rates (division-level, with global fallback)

we_rate(c) = SUM(c) / NULLIF(SUM(t), @)
global_rate(c) = SUM(c) / NULLIF(SUM(t), 0)

Posterior and smoothed count

Apost = (we_rate #1 +X) / (t+ t)
X_smooth = t * A_post

27/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
Pseudo-minutes t (global defaults used)

* Striking: sig str = 0.7, head = 0.8, body = 2.5, leg = 2.1

* Grappling: td = 7.0, sub = 12.0, kd = 20.0, rev = 42.0

* Round 1: sig_str_rd1 = 0.7, head_rd1 = @.7, body_rd1 = 2.5, leg_rdl = 1.7, td_rdl =
9.0, sub_rdi = 15.0, kd_rdi = 12.0, rev_rdi = 60.0

Per-weight-class overrides (only where cross-validation helped)

+ Flyweight: rev = 22.0 (vs 42.0 global)
* Light Heavyweight: head_rdi = 0.5 (vs 0.7)
+ Heavyweight: td = 5.0 (vs7.0), td_rdi = 4.0 (vs 9.0)

Execution notes

+ Emit c_smooth values with a _smooth suffix; then rename so smoothed values replace original:
Keep c_raw only long enough to compute totals/accuracy/defense/ratios/per; then drop it.
* Filter prior CTEs to rows with time_sec > 0.

3) Beta-Binomial Smoothing (Binary + Control Share)

Families and attempts

* ko, win, decision : successes = 0/1; attempts = 1 per fight.

*  sub_land : successes = sub_land ; attempts= sub_att .

* ctrl, ctrl_rd1 (duration as share): attempts = seconds ( min(rd1,300) for rd1); we output
smoothed seconds = p_post x attempts -

Posterior mean

p_post = (rate_prior * t + successes) / (t + attempts)
if attempts = 0: p_post = rate_prior

Pseudo-counts Tt (global defaults used)

* Global: ko = 23, win = 25, decision = 20, sub_land = 9, ctrl = 2

© Round 1: ko_rdl = 17, win_rdl = 15, decision_rd1 = 16, sub_land_rdl = 7, ctrl_rdi =1

Per-weight-class overrides (data-validated)

* Featherweight: sub_land = 3 (vs9)
* Light Heavyweight and Heavyweight: ctrl = 1.5 (vs 2.0)

Priors

we_rate = SUM(successes) / NULLIF(SUM(attempts), 0)
global_rate as fallback

Execution notes

+ Run Beta-Binomial before Poisson-Gamma so attempts like sub_att are unmodified when you
need them.
* Store + smooth ; then participate in the same rename step as counts.

4) Opponent Features and Priors (plumbing that makes AdjPerf work)

1. Opponent "allowed" stats: compute what other fighters did against each fighter (this is the
"personal history" the opponent carries into a matchup).

2. Weight-class aggregates: build per-feature _wc_mean, _wc_mad, and _minimum_mad tables.
These are used as priors and floors in AdjPerf.

3. Strict time ordering: when joining history to a current fight, only include rows strictly earlier the
the current fight's event date (with tie-breakers).

28/79


1/6/26, 9:08 AM MMA-ALnet

5) Minimal SQL/Pseudocode you can adapt

Opponent history for a column c (decay optional)

rows_c AS (

SELECT cur.fight_id, cur.fighter_id, hist_opp.c AS val,

CASE WHEN :decay THEN EXP(-lambda * age_years) ELSE 1.0 END AS w

FROM features.<table> cur

JOIN fight_mapping fm_cur ON cur.fight_id = fm_cur.fight_id

JOIN event_mapping em_cur ON fm_cur.event_id = em_cur.event_id

figure out opponent id for the current fight

join to all past fights where others faced that opponent

-- restrict to strictly earlier fights (event_date, event_id, fight_id)
)

med_c AS (

SELECT fight_id, fighter_id,

PERCENTILE_CONT(@.5) WITHIN GROUP (ORDER BY val) AS med

FROM rows_c WHERE val IS NOT NULL GROUP BY fight_id, fighter_id

)

stats_c AS (

SELECT fight_id, fighter_id,

SUM(weval) / NULLIF(SUM(CASE WHEN val IS NOT NULL THEN w END),0) AS c_opp_mean_pers
FROM rows_c GROUP BY fight_id, fighter_id

)

mad_e AS (

SELECT fight_id, fighter_id,

PERCENTILE_CONT(@.5) WITHIN GROUP (ORDER BY ABS(val - med)) AS c_opp_mad_pers
FROM rows_c JOIN med_c USING(fight_id, fighter_id)

GROUP BY fight_id, fighter_id

)

n_hist AS (

SELECT fight_id, fighter_id,

CASE WHEN :decay THEN POWER(SUM(w),2)/NULLIF(SUM(POWER(w,2)),@) ELSE COUNT(*) END AS n
FROM rows_c GROUP BY fight_id, fighter_id

)

Adjusted Performance scoring

-- inputs: observed c, n, c_opp_mean_pers, c_opp_mad_pers, c_wc_mean, c_wc_mad, ¢_mad_floor
wmean =n / (n + Kmean)

wimad =n / (n + K mad)

mu = w_mean + c_opp_mean_pers + (1 - w_mean) * ¢_wc_mean

sigma = GREATEST(w_mad * ¢_opp_mad_pers + (1 - w_mad) * c_wc_mad, c_mad_floor)

score = GREATEST(LEAST((observed - mu) / sigma, 7.0), -7.0)

Count smoothing (Poisson-Gamma)

t = CASE WHEN is_rd1 THEN LEAST(time_sec_rd1,300)/60.0 ELSE time_sec/60.0 END
we_rate = SUM(c)/SUM(t) (per weight class; have a global fallback)
X_smooth = t * ((we_rate # t +X) / (1 + t))

Binary smoothing (Beta-Binomial)

n= attempts -- the number of attempts for that row
Xx = successes — the observed successes for that row
rate_p = we_rate or global_rate if we missing

p_post = (rate_p *t +x) / (t +n)

output = p_post -- or p_post * n for control seconds

6) Practical QA checks

* Distribution sanity: after smoothing, per-minute rates and binary probabilities should cluster
around WC priors for tiny n and drift toward personal rates as n grows.

https://www.mma-ai.net/news 29/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
* Leakage audit: for a random sample of fights, verify that no future rows contribute to opponent-
history or priors.
+ Sensitivity: n=0,1,2 cases should produce conservative AdjPerf magnitudes; heavy shrink on
dispersion reduces fake "z-score" spikes.

7) The Reality Check: Performance vs. Expectations

After all this technical work—the opponent-aware adjusted performance, the Poisson-Gamma smoothin
the Beta-Binomial calibration—you might expect dramatic improvements in accuracy or log-loss. The
reality is more sobering.

We're still hitting roughly 70% accuracy on the unseen test set (2024-2025 fights), with log-loss
numbers virtually identical to previous model versions. The fundamental limitation isn't in our statistical
methodology—it's in our feature set.

What Actually Improved

The main advantage of v7 is better calibration. The model's confidence scores now align more closely
with actual win frequencies. This means:

* When the model is wrong, it tends to be less confident about those incorrect picks
* We can be more selective about +EV betting opportunities, leading to slightly better RO!
* The calibration curve shows the model is less overconfident in the 60-80% probability range

The Feature Ceiling Problem

We're fundamentally limited by using the same base statistics that every MMA model relies on: strikes
landedjattempted, takedowns, control time, etc. These are the stats that have been available consistent
across the last 10+ years of UFC data, but they only capture so much of what determines fight outcome

Real improvements in accuracy would require fundamentally new features:

Sentiment analysis from fighter interviews, social media, and press conferences

Meta-analysis of other successful handicappers and their picking patterns
Contextual factors like training camp disruptions, weight cuts, personal circumstances

Video analysis of technique, footwork, and tactical tendencies

The problem is consistency—these features are either unavailable for historical fights or extremely diffic
to collect reliably across thousands of matchups.

The LLM Temptation (and Its Limitations)

Large Language Models offer an intriguing possibility for incorporating qualitative analysis, but they
present a fundamental backtesting problem. You can't tell an LLM to "forget everything that happened
after 2022" to create a proper train/test split. Without the ability to backtest on truly unseen data, it's
impossible to know if LLM-enhanced predictions would actually generalize or just memorize recent fight
outcomes.

This creates a catch-22: the most promising avenues for improvement (LLMs, real-time sentiment, insid
information) are exactly the ones that break our ability to validate models scientifically.

Moving Forward

V7 represents the practical ceiling for what's achievable with traditional fight statistics and rigorous
backtesting methodology. The technical improvements here—better smoothing, opponent-aware featur:
proper calibration—are the difference between a good model and a great one, even if that difference is
measured in percentage points rather than dramatic leaps.

Future breakthroughs will likely come from entirely new data sources rather than more sophisticated
processing of the same old numbers. Until then, we'll continue refining the edges and squeezing every t
of signal from the statistics we have.

30/79


1/6/26, 9:08 AM MMA-ALnet

https://www.mma-ai.net/news 31/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Data Drift, Generalization, and the Quest for ¢
Bulletproof UFC Model

January 20, 2025

One of the hardest parts of training a sports model on UFC is the data drift. In 2016 the Vegas odds wer
only 61% accurate, then in 2024 they were something like 70% accurate. Right now we're seeing a smal
decline in the accuracy of Vegas. This is all just part of the ebb and flow of variance that also factors in
general unpredictability of sports, especially UFC.

LEAST (GREATEST( (observed_strikes - (oh.n_eff_strikes/(oh.n_eff_strikes + K_mean_strikes) *
oh.opp_strikes_mean + (K_mean_strikes/(oh.n_eff_strikes + K_mean_strikes)) * we.strikes_mean)) /
GREATEST (oh.n_eff_strikes/(oh.n_eff_strikes + K_mad_strikes) * oh.opp_strikes_mad +
(K_mad_strikes/(oh.n_eff_strikes + K_mad_strikes)) * we.strikes_mad, mad_floor), -7), 7) ELSE
(observed_strikes - we.strikes_mean) / GREATEST(we.strikes_mad, mad_floor) END AS strikes_adjperf,
Similar pattern for grappling with grappling-specific parameters CASE WHEN oh.n_eff_grappling >= 1
THEN -- Use grappling-specific K values and opponent history ELSE -- Fall back to weight-class only Et
AS grappling_adjperf FROM fight_stats fs LEFT JOIN opponent_history_per_column oh USING (fight_id
fighter_id) JOIN weight_class_priors we ON fs.weightclass = we.weightclass ) -- Typical K values | use: -
K_mean_strikes = 8, K_mad_strikes = 12 -- K_mean_grappling = 5, K_mad_grappling = 8 -- Higher
K_mad provides more stability for variance estimates

Critical implementation insights:

Per-column effective n: Compute per-column n_eff and set weights to zero when that column hi
no opponent history, instead of COALESCE(..., 0) pretending it does.

Global priors as safety net: Always have a global fallback row in weight-class prior CTEs so sparse
classes never collapse toward zero.

Per-stat K tuning: Tune K_mean VS K_mad per stat family. Generally K_mad > K_mean for stability
with small samples.

Time decay implementation: Use EXxP(-A + years_ago) with A = 0.13 for moderate decay over ~5
year half-life.

2) Poisson-Gamma Smoothing (Counts)

What was wrong before

Under-specified exposure: | mixed count smoothing and rate logic inconsistently. The posterior
mean was formed from (mean, variance) but | didn't consistently convert through exposure time wh
mapping back to counts. Short fights could look "low activity" for the wrong reason.

+ Ad-hoc variance branch: When var < mean |used a hand-rolled blend (e.g., (priors3 +
observed) /4 ). It stabilized things but it was a heuristic—not a model.

One-size priors by year slice: Priors came from a fixed date window with mean/variance; no explic
rate per minute, and no principled pseudo-exposure strength.

Binary/duration leakage risk: The old pass didn't clearly wall off duration (control time) or binary
signals from count smoothing; at best they were excluded by name heuristics.

What I do now

32/79


1/6/26, 9:08 AM MMA-ALnet

+ Rate-based Bayesian updating: | compute weight-class rates per minute ( u_w ) and update with
exposure time t (minutes). The posterior is A_post = (uw * t +X) / (t + t),andI map backt
counts via X_smooth = t * A_post .

* Validated pseudo-minutes: + (the prior strength) is per stat and sometimes per weight class (on
when cross-validation showed 20.5% lift). Otherwise | fall back to global + for consistency.

+ Explicit exposure rules: Round-1 columns use min(time_sec_rd1, 300)/60 ; everything else uses
‘time_sec/60 . No more implicit exposure guessing.

* Strict scope: Count data only (e.g., *_land, *_att, kd, rev). Binary and duration stats are handled
elsewhere.

Implementation Details for Your Own Model

Here's the specific SQL pattern | use for Poisson-Gamma smoothing:

-- Step 1: Compute weight-class rate priors
WITH we_priors AS (
SELECT weightclass,
SUM(stat_count) / NULLIF(SUM(time_sec / 60.0), @) AS wc_rate
FROM fight_stats
WHERE time_sec > 0
GROUP BY weightclass

-- Step 2: Apply Poisson-Gamma smoothing
smoothed AS (
SELECT fight_id, fighter_id,
-- Posterior rate: (prior_rate * pseudo_minutes + observed_count) / (pseudo_minutes
(p.we_rate * 1 + fs.stat_count) / (t + fs.time_sec / 60.0) AS posterior_rate,
-- Convert back to smoothed count
(fs.time_sec / 60.0) *
((p.we_rate * t + fs.stat_count) / (t + fs.time_sec / 60.0)) AS stat_count_smooth
FROM fight_stats fs
JOIN we_priors p ON fs.weightclass = p.weightclass

-- Key insight: t values I use:
Striking stat: 15-25 minutes
Grappling stats: t = 8-12 minutes
-- Submission attempts: t = 5-8 minutes

Critical implementation notes:

Filter out zero exposure: Always add WHERE time_sec > @ when computing priors to avoid divisior
by zero.

+ Round-1 exposure cap: For first-round stats, cap exposure at 300 seconds (5 minutes) since roun
can't exceed this.

+ Pseudo-minute tuning: Start with t=15 for most stats, then use cross-validation to optimize. Highe
‘T= more shrinkage toward prior.

* Weight-class fallbacks: Always have a global prior for weight classes with insufficient data.
Why this fixes the old issues

* Short/long fights handled correctly: A two-minute brawl vs. a fifteen-minute chess match are nov
comparable because smoothing happens on rates and returns properly scaled counts.

* Less hand-waving, more model: Pseudo-minutes encode prior confidence without ad-hoc
branches; per-class overrides exist only where they earned their keep.

https://www.mma-ai.net/news 33/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
3) Beta-

inomial Smoothing (Binary)

What was wrong before

+ There wasn't any. KOWin/Decision/Sub were effectively raw indicators or simple ratios. That's nois
mis-calibrated, and conflates "skill" with "opportunity."

* Attempts were undefined: | had no principled "denominator" for success probability. For example,
calling every strike a KO attempt is just wrong.

What I do now
* Proper Beta-Binomial: For each binary family | define attempts and successes:

© ko/win/decision : each fight is one opportunity.
© sub_land : opportunities = sub_att .

© ctrl (duration): modeled as a time-share (Bernoulli per second); smoothed rate times exposu
seconds.

+ Weight-class priors, validated strength: | use WC success rates as p priors and add pseudo-
counts + tuned per stat, optionally per WC where cross-validation justifies it (e.g., Featherweight
subs, LHW/HW control).

+ Zero-attempt guard: If attempts are zero, | return the WC/global prior rate rather than fabricating ¢
fraction.

Implementation Details for Beta-Binomial Smoothing

Here's the exact approach for different binary outcomes:

-- For KO/Win/Decision rates (attempts = 1 per fight)
WITH we_binary_priors AS (
SELECT weightclass,
SUM(CASE WHEN outcome = 'ko' THEN 1 ELSE 0 END)::float / COUNT(+) AS ko_rate,
SUM(CASE WHEN outcome = 'win' THEN 1 ELSE @ END)::float / COUNT(*) AS win_rate
FROM fight_results
GROUP BY weightclass
yy

-- For submission success rates (attempts = sub_att, successes = sub_land)
sub_priors AS (
SELECT weightclass,
SUM(sub_land)
FROM fight_stats
WHERE sub_att > 0
GROUP BY weightclass

‘loat / NULLIF(SUM(sub_att), @) AS sub_success_rate

yy

-- Apply Beta-Binomial smoothing
smoothed_binaries AS (
SELECT fight_id, fighter_id,
-- KO rate: (prior_rate * pseudo_attempts + successes) / (pseudo_attempts + attempt
(p.ko_rate * t_ko + ko_success) / (t_ko + 1) AS ko_prob_smooth,
-- Sub rate: (prior_rate * pseudo_attempts + sub_land) / (pseudo_attempts + sub_att
(sp.sub_success_rate * t_sub + sub_land) / (t_sub + GREATEST(sub_att, 1)) AS sub_pt
FROM fight_stats fs
JOIN we_binary priors p ON fs.weightclass = p.weightclass
JOIN sub_priors sp ON fs.weightclass = sp.weightclass

-- Pseudo-attempt values I use:
-- KO/Win/Decision: t = 8-12 fights

34/79


1/6/26, 9:08 AM MMA-ALnet

-- Submission success: t

3-5 attempts

-- Control time share: t = 600-900 seconds

Key implementation considerations:

+ Zero-attempt handii
to compute a ratio.

g: When sub_att = 0, use the weight-class prior rate directly rather than tryir

* Control time as time-share: Model control duration as Bernoulli per second, then multiply smooth
probability by total seconds.

* Minimum attempts: Use GREATEST(attempts, 1) to avoid division by zero in edge cases.

* Cross-validation tuning: Start with t values above, then optimize per weight class only if CV show
>0.5% improvement.

Why this fixes the old issues

* Calibration: Probabilities stop over- or under-shooting because we borrow signal from the division
realistically.

* Correct denominators: A sub success isn't "one per fight;" it's "out of sub attempts." A KO isn't "f
strike;" it's "per fight." This matters.

* Principled uncertainty: Small sample sizes get more shrinkage, large samples trust their own data
more.

4) Pipeline & Naming Hygiene

+ Order matters: Binary smoothing runs before count smoothing (so sub attempts are raw counts
when they need to be). After smoothing, | temporarily keep *_raw , compute derived stats (totals,
accuracy, defense, ratios, per), then | drop +_raw . AdjPerf waits until opponent and weight-class
aggregates are ready.

* Consistent feature families: Head/Body/Leg and Distance/Clinch/Ground share a naming pattern,
per/ratio/opp calculations can be reliably applied and diffed.

Why this fixes the old issues

+ No accidental toggling: Derived features don't unknowingly mix raw and smoothed values.

+ Less glue code, fewer footguns: Calculators can target families by suffix/prefix without bespoke
filters for every one-off.

Complete Implementation Roadmap

If you're building your own MMA prediction model, here's the exact order of operations that prevents da
leakage and ensures all derived features use properly smoothed inputs:

Step-by-Step Pipeline
1. Base feature extraction: Raw fight stats > fight_stats_derived table
2. Beta-Binomial smoothing: Binary outcomes (KO, win, decision, sub success, control time-sha
Poisson-Gamma smoothing: Count statistics (strikes, takedowns, etc.)
Temporary raw preservation: Keep +_raw columns during derivation phase
Derived feature computation: Totals, accuracy, defense rates, ratios - all using smoothed valu

Raw column cleanup: Drop +_raw columns after derived features are complete

NOOB w

Per-minute and ratio features: Rates, pressure metrics, position-specific stats

https://www.mma-ai.net/news 35/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

8. Feature family tables: Separate tables for striking, grappling, position stats
9. Opponent aggregation: Build opponent history tables with time decay
10. Weight-class priors: Compute means, MADs, and minimum MAD floors

11. Adjusted Performance: Apply opponent-aware, reliability-weighted standardization

Key SQL Patterns You Can Reuse

-- 1. Time decay weight calculation
SELECT EXP(-0.13 + EXTRACT(years FROM current_date - fight_date)) AS decay weight

-- 2. Effective sample size (Kish formula)
SELECT POWER(SUM(w), 2) / NULLIF(SUM(POWER(w, 2)), @) AS n_effective

-- 3. Shrinkage toward prior
SELECT (n/(n + K) * observed + K/(n + K) * prior) AS shrunk_estimate

-- 4. Robust MAD calculation
WITH medians AS (
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY stat_value) AS median_val
FROM stats_table
yy
deviations AS (
SELECT ABS(stat_value ~ m.median_val) AS abs_dev
FROM stats_table s CROSS JOIN medians m
)
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY abs_dev) AS mad_value
FROM deviations

-- 5. Exposure-corrected rate calculation
SELECT stat_count / GREATEST(time_minutes, 0.1) AS rate_per_minute

Hyperparameters That Matter Most

Time decay A: 0.13 gives ~5 year half-life for fight relevance

Poisson-Gamma t (pseudo-minutes): 15-25 for striking, 8-12 for grappling

Beta-Binomial t (pseudo-attempts): 8-12 for outcomes, 3-5 for submission success

Shrinkage K values: K_mean = 5-8, K_mad = 8-12 (higher for stability)

Adjusted performance clipping: [-7, +7] to prevent extreme outliers

Common Pitfalls to Avoid

Data leakage: Always filter opponent history to event_date < current_fight_date

Zero exposure: Add WHERE time_sec > @ filters when computing rate priors

Missing weight classes: Include global fallback rows in all prior computations

Mixed raw/smoothed: Ensure derived features use smoothed inputs consistently

Inadequate sample sizes: Set minimum thresholds for opponent history reliability

36/79


1/6/26, 9:08 AM MMA-ALnet

Data Drift, Generalization, and the Quest for ¢
Bulletproof UFC Model

January 20, 2025

One of the hardest parts of training a sports model on UFC is the data drift. In 2016 the Vegas odds wer
only 61% accurate, then in 2024 they were something like 70% accurate. Right now we're seeing a smal
decline in the accuracy of Vegas. This is all just part of the ebb and flow of variance that also factors in
general unpredictability of sports, especially UFC.

For the latest version of the model | spent a ton of time trying to really generalize the model so that I'm r
overfitting to specific circumstances at any small point in time. These are the parameters | ended up
settling on after hundreds of experiments:

train_size = 0.75
val_size = 0.15

test_size = 0.1

nsplits = 4
num_stack_levels = 2
use_recency_weights = True
use_bag_holdout = True # Must be true if we're using tuning data (val split)
num_bag_sets = 2
decay_rate = 0.13

shuffle = True

start_date = '2014-04-01'
calibrate = True

Breaking Down the AutoGluon Parameters

Let me explain each of these settings in simple terms:

* train_size/val_size/test_size: We use 75% for training, 15% for calibration validation, and 10% for

final testing—targer validation than test because calibration requires substantial data to work prope!
* n_splits: Cross-fold validation uses 4 splits on the training data to ensure robust model selection.

+ num_stack_levels: AutoGluon stacks models in 2 layers, where second-layer models learn from
first-layer predictions.

* use_recency_weights: More recent fights are weighted heavier in training to capture current fighti
trends.

+ decay_rate: At 0.13, the earliest fight from 2014 weighs about 0.5 while the latest fight weighs abot
1.5—making recent fights 3x more important.

* use_bag_holdout/num_bag_sets: Creates multiple model bags with holdout validation for better
ensemble diversity.

+ shuffle: Randomizes training order to prevent temporal bias during model training.

* calibrate: Applies post-hoc probability calibration to improve confidence score reliability.
For a long time | was doing a Cross-Fold Validation on 90% of the data, then the last 10% (about 1 year «

the most recent fights) was left unseen as an impartial test of how well the model is generalizing to
unseen future fights.

https://www.mma-ai.net/news 37/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

4-fold validation (k=4)

Fold 1 Testing set 4
Fold 2 Training set Testing set Training set 4

Fold 3 Training set Testing set Training set 4

Fold 4 Training set | Testing set 4

Cross-validation ensures robust model evaluation by training and testing on different data splits, helping preven
overfitting and providing more reliable performance estimates.

The Evolution of Vi

lation Strategy

The reason for this was | wanted to maximize the most recent fight data to better predict near future
results. As the years have gone by and I've gained a better overhead view of general variance in this spo
\'m slightly leaning towards really thinking through the architecture with generalization in mind will
perform better over the long term rather than trying to maximize training data to the most recent fights.
But like everything else in this journey, I'll experiment with this strategy and possibly revert back
depending on how things go in the testing and real world as time goes on.

Once | started messing with calibration again | was required to go back to a train/val/test method where
we do CFV on the train set then calibrate the predictions on val and get an unbiased evaluation of the
model from unseen data on test.

So 2014-2023 is training, 2024 is calibration, and 2025 is test (more or less). Over hundreds of tweaks 1
the parameters above, this final set of parameters is showing the most consistency. I'm very sensitive to
not just use the model that had the best results on the validation and test results because that can lead

overfitting where we just tuned parameters to maximize performance on the training and validation sets
Instead, we make sure that there isn't too much difference between validation and test set performance
and that the training set performance isn't wildly higher than val/test. Results:

Training accuracy: 0.7511
Training log loss: -0.5082
Test accuracy: 0.7008
Test log loss: -0.5949
Val accuracy: 0.7072

Val log loss: -0.5918

Very nice sub -0.6 logloss and >70% accuracy on both the validation set and the unseen test set. Just tc
make sure we're not overfitting the model | started doing experiments where | use the same parameters
and train the model based on cutoff dates, so how would these parameters perform if it didn't have
access to the last 6 months of data or the last 12?

The P-Hacking Trap

Same parameters using data with a cutoff of 6 months ago:

38/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Training accuracy: 0.7874
Training log loss: -0.4849
Test accuracy: 0.7149
Test log loss: -0.5785
Val accuracy: 0.6542

Val log loss: -0.6044

Reasonable. So | feel pretty good about the generalizability of the parameters I'm using right now and th
training method implemented. | can p-hack the shit out of this though. For those unfamiliar, p-hacking is
term used in statistics where you "massage" the data or process so it looks like you're measuring some
metric in a reasonable way but in fact you're just tweaking small portion of the stats so that the final
benchmark metric like accuracy or logloss is maximized to make yourself look smart but isn't accurate t
real-world measurement. For instance, what happens if | make a tiny change like increasing the initial da
cutoff by 4 months from 2014-01-01 to 2014-04-01?

Training accuracy: 0.8460
Training log loss: -0.4122
Val accuracy: 0.7056

Val log loss: -0.5895
Test accuracy: 0.7121
Test log loss: -0.5864

Wowie weewah | improved the validation and test set logloss AND accuracy! I'm a motherfucking genius
This kind of improvement is unlikely to generalize to real world increases of accuracy and logloss. Look «
the fact that the training accuracy increased by 6% leading to a larger gap between the training
accuracy/logloss and the val/test logloss/accuracy. This is a negative improvement that likely means the
model is a little bit more overfitted to the historical data.

You could argue this isn't that relevant since the unseen fight data accuracy and logloss improved and y
wouldn't necessarily be wrong, but here in lies the difficulty of machine learning: you're always
backtesting and there is no way to look into the future and tell how the model will actually perform in the
real world, all you can do is decipher clues and the clue that stands out to me is a giant 6% leap in trainin
data accuracy lead to an almost 15% gap between training accuracy and val/test accuracy. This is not
good because we will never see 85% accuracy in unseen data, ever. We are straying further from Jesus
with this minor change. We could've just eliminated a chunk of high variance fights where a bunch of
unpredicted big underdogs won but by eliminating these high variance fights, we might've harmed the
model's ability to recognize patterns in high variance fights that are still likely to occur in the future and :
harmed the generalized ability of the model to predict future fights.

Why Not Include Odds as Features Anymore?

Because the odds accuracy vary wildly compared to more predictable sports like MLB, Soccer, or NFL.
Again, the odds were 61% accurate in 2016 yet 70% accurate in 2024. Including the odds essentially
makes the model subject to the whims of vegas rather than concretely generalizable over the long term.
Second, since we have such highly engineered features, including the odds barely increases the accura
of the model although it does improve the logloss quite well. All my odd-included models are -ROI on all
down-the-line Al predictions because it so heavily favors betting the vegas odds favorite. It is useful still
as a secondary measurement of risk-adjusted returns on favorites, but ultimately it's not that useful in
generalized risk-adjusted returns on predictions.

Betting Strategy Performance Analysis

The real test of any model isn't just accuracy—it's profitability. Here's how different betting strategies
performed over the latest backtest period, starting with $1,000 and betting $10 per pick:

Key Performance Metrics Explained

* ROI (%): Return on investment, measuring profit as a percentage of total amount wagered.

39/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Sharpe (ann.): Risk-adjusted return that accounts for volatility—higher values indicate better risk-
adjusted performance.

Sortino (ann.): Similar to Sharpe but only penalizes downside volatility, focusing on harmful risk.

CAGR (%): Compound Annual Growth Rate, showing how fast your bankroll would grow annually.

Max DD (%): Maximum drawdown, the largest peak-to-trough decline in bankroll value.

Calmar: CAGR divided by maximum drawdown, measuring return per unit of downside risk.

PF: Profit Factor, the ratio of gross profits to gross losses.

ROI-Sharpe: A custom metric combining ROI and Sharpe ratio for overall strategy evaluation.

Backtest Summary (2024-08-03 to Present)

Test Period: Starting from August 3, 2024 with $1,000 initial bankroll and $10 bet size

‘Y Best Overall: ai_all_picks_sevenday hil Closing Odds Strategy
ROI: 10.87% | Sharpe: 2.11 | Final Bankroll: ROI: 9.51% | Sharpe: 1.83 | Final Bankroll:
$1,287.02 $1,250.98

@ Edge Threshold Strategy

ROI: 3.68% | Sharpe: 0.47 | Final Bankroll:
$1,086.01

Complete Performance Data: For detailed metrics including CAGR, maximum drawdown, Sortino
ratios, and more, download the full backtest results:

© Download Complete Backtest Data (TXT)

+ Download Complete Backtest Data (CSV)

Model Calibration Analysis

Beyond profitability metrics, it's crucial to understand how well our Al model's confidence levels align wi
actual outcomes. A calibration curve shows whether the model's predicted probabilities match real-wor
frequencies—if the Al says a fighter has a 70% chance of winning, do they actually win about 70% of the

time?

40/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Calibration Curve

104 --- Perfect Calibration of
“a+ Model (Test)
-e- Vegas (Test)

08

06

oa

Fraction of Positives

02

0.0 4+"

00 02 o4 os oa 10
Mean Predicted Probability

Calibration curve showing the relationship between predicted probabilities and actual outcomes. A perfectly
calibrated mode! would follow the diagonal line—deviations indicate overconfidence (above line) or
underconfidence (below line) in predictions,

The calibration curve reveals how reliable our model's confidence estimates are across different
probability ranges. This is essential for betting strategies because misaligned confidence can lead to po
risk assessment—betting too heavily on picks the model is overconfident about, or missing value when t
model is underconfident.

Key Insights from the Strategy Analysis

Lines generally move against us: The fact that the seven-day strategy consistently outperforms closir
odds betting is a GREAT sign that the model is a sharp picker. When books adjust lines toward our picks
by fight night, it validates our edge.

Risk-adjusted returns matter most: For gamblers, the most important metric isn't ROI, accuracy,
logloss, or even expected value alone—it's risk-adjusted returns. You want sustainable profit without
excessive volatility that can bankrupt you during inevitable losing streaks.

Based on the comprehensive analysis, the optimal strategy appears to be betting all Al picks within a 16
difference between the Al win probability and Vegas implied probability. For example, if the Al picks
fighter! at 70% and Vegas implied odds suggest 84%, we still bet on fighter1. Beyond this 16% threshol
we should consider placing half-unit bets AGAINST the fighter the Al picked.

Risk-Adjusted Strategy Performance

First, thanks to the community for helping me learn here (Shoutout to Jordan C. and the rest of you). To
better understand the relationship between risk and reward across different betting strategies, we've
analyzed the top 20 performing strategies based on their risk-adjusted returns. The scatter plot below
shows each strategy's return on investment (ROI) versus its Sharpe ratio, which measures risk-adjusted
performance.

41/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Top 20 Strategies: Risk-Adjusted Performance
(Size = Total Profit Magnitude, Color = Sharpe Ratio)

Pratt Mogie FELD)
500 po
52500 pot
$3000 prot
oss
a) a
a
2 Sore)
§ °
Jon
2 20
é 1
5 oso
°
°
1
a8 °
6 °
e ° wr
0.78 °
3 i 7 Pe 2 3 3

Return on Investment (RO!) %

Scatter plot showing ROI vs Sharpe ratio for the top 20 betting strategies. Strategies in the upper-right quadran
offer the best combination of high returns and good risk-adjusted performance.

The ideal strategies appear in the upper-right quadrant, offering both high ROI and strong risk-adjusted
returns (high Sharpe ratio). This visualization helps identify strategies that not only generate profit but d
so with manageable risk and volatility. Most of the "bet against Al pick based on (Al win% - Vegas odds
implied probability) strategies have a cutoff of about 16-20%. Meaning the right time to bet against the .
profitably is when Al win% is at least 16-20% lower than the Vegas implied odds difference and even the
it's probably not worth more than 1/2 a unit. The model appears to be very good at predicting underdog:
at about a 50% rate, but it's general ability to accurately predict win% is still not as good as Vegas
meaning betting solely on +EV isn't necessarily the best strategy.

Conclusion and Call for Feedback

Two important points:

1) Please be critical of this and send me unfiltered opinions and help. | have nowhere near the level «
experience in picking betting strategies as | do in training models. The machine learning side I've got
down, but translating statistical edges into optimal betting strategies is where | need the most
improvement.

2) The data suggests our best approach is the nuanced edge-threshold strategy rather than blindly
following all Al picks or only +£V selections. This makes intuitive sense—we trust the model most when i
aligns reasonably well with market expectations, but we hedge when there's significant disagreement.

As always, this is an ongoing experiment. The beauty and curse of sports prediction is that the landscap
constantly shifts. What works today might not work tomorrow, which is why building robust, generalizab
systems matters more than chasing short-term optimization.

Get in Touch: The best place to reach me is on Patreon where the community shares free UFC
predictions and discuss model development. You don't need to subscribe to participate in chats and
direct messages—it's the most active community for discussing these predictions and sharing feedback
I'd love to hear your thoughts on the strategies and any suggestions for improvement!

42/79


1/6/26, 9:08 AM MMA-ALnet

https://www.mma-ai.net/news 43/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Calibrating UFC Fight Predictions

July 15, 2025

Introduction

My UFC prediction model has maintained profitable performance for years despite having poorly
calibrated probability estimates. While the model excels at binary classification (picking winners), its
confidence scores don't align well with real-world win frequencies—a classic case of good accuracy, bat
calibration.

This never bothered me much since the model was profitable, but | decided to experiment with post-hoc
calibration methods to see if | could improve probability estimates without hurting classification
performance. This post documents those experiments: the methods | tested, why most failed with limite
data, and the modest improvements | eventually achieved with Platt scaling.

What is Model Calibration?

Model calibration refers to how well a model's predicted probabilities align with actual observed
frequencies. A perfectly calibrated model should be correct 70% of the time when it predicts a 70%
probability, 80% of the time when it predicts 80%, and so on.

Consider this example: if your model predicts 100 fights at 60% confidence, and the favored fighter win
60 of those fights, your model is well-calibrated at that confidence level. However, if the favored fighter
wins 75 times, your model is under-confident and poorly calibrated.

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Example: plotting calibration curve
fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
plt.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
plt.plot(mean_pred, fraction_pos, 's-', label='Model')
plt.xlabel('Mean Predicted Probability’)

plt.ylabel( ‘Fraction of Positives')

The Calibration vs. Accuracy Paradox

One of the most counterintuitive aspects of calibration is that improving probability estimates can
sometimes hurt classification accuracy. This happens because calibration optimizes for probability-base
metrics like log-loss and Brier score, while accuracy only cares about the binary decision boundary at
50%.

Why This Happens

Consider a model that consistently predicts 65% when the true probability is 60%. This model might
achieve higher accuracy by being overconfident (correctly picking the winner more often due to increas
confidence), but it's poorly calibrated. When calibrated, the model's predictions become more
conservative, potentially crossing the 50% decision boundary in some cases and reducing accuracy.

In my UFC model, this manifested as:

Original Model Performance:
Accuracy: 0.7098

44/79


1/6/26, 9:08 AM MMA-ALnet

Log Loss: 0.6032
Brier Score: 0.2075

Calibrated Model Performance:
Accuracy: 0.7098 (unchanged)

Log Loss: 0.5979 (improved by 0.0053)
Brier Score: 0.2056 (improved by 0.0019)

The accuracy remained stable while probability-based metrics improved significantly but that isn't true {
all calibration methods and isn't even necessarily true 100% of the time for the method that actually
worked.

The Small Sample Size Challenge

My dataset contains approximately 2,400 UFC fights over 10 years after extensive filtering:

def filter_fights(df, threshold, dat.

2015-01-01", include_split_de:

Filter fights based on:
Binary results (y_true in [0, 11)

Both fighters must have had at least num_fights previous fights
Removing unwanted fight methods (DQ, split decisions, etc.)
Fights from 2015 onward

# Remove unwanted methods
if include_split_dec:

unwanted_methods
else:

['dq', ‘other’, ‘overturned']

unwanted_methods = ['dq', ‘other', ‘decision - split', ‘decision - majority', ‘overtut

# Filter to only binary results and recent fights
f = df[df['y_true'].isin([0, 1])].copy()
f = df[df['event_date'] >= pd.Timestamp(date) ]

return df

This small sample size created significant challenges for calibration, particularly with isotonic regression
which requires sufficient data points across the probability spectrum.

Isotonic Regression: The First Attempt

Isotonic regression is a non-parametric calibration method that learns a monotonic mapping from
predicted probabilities to calibrated probabilities. It's theoretically superior to Platt scaling as it can
capture non-linear calibration relationships.

class SimpleTsotonicCalibration:
def _init_(self, y_min=0.01, y_max=0.99):
self.y_min = y_min
self.y_max = y_max
self.calibrator = None

def fit(self, y_prob, y_true):
from sklearn. isotonic import IsotonicRegression

self.calibrator = IsotonicRegression(
y_min=self.y_min,
y_max=self.y_max,
out_of_bounds="clip!

)
self.calibrator.fit(y_prob, y_true)

Isotonic Results (Disappointing)

https://www.mma-ai.net/news 45/79


1/6/26, 9:08 AM MMA-ALnet

Isotonic Calibration Results:
Should Use Calibration: False

Calibration Set Log Loss Improvement: 0.0553 # Improvement on the calibration set of data
Test Set Log Loss Improvement: 0.0125 # Worse on the unseen test set!

Test Set Brier Score Improvement: 0.0023 # Worse on the unseen test set!

Final Test Log Loss (Original): 0.5937

Final Test Log Loss (Calibrated): 0.6063

The isotonic calibration actually hurt performance on the test set despite improving calibration set
metrics. This is a classic sign of overfitting due to insufficient data.

Calibration Curve

1.04 —7—- Perfect calibration
—@ Original
—®- Isotonic calibrated
0.8
9
$
206
rd
é
5
<
2
0 04
je
c
0.2
-
0.0
0.0 0.2 0.4 0.6 0.8 1.0
Mean Predicted Probability
Why Isotonic Failed

1. Small sample size: With only ~240 test samples (10% of 2,400), isotonic regression had
insufficient data to learn a robust monotonic mapping

2. Sparse probability regions: Some probability ranges had very few examples, leading to
unreliable calibration

3. Overfitting: The flexibility of isotonic regression became a liability with limited data

| attempted several improvements:
+ Ensemble approach: Multiple isotonic regressors trained on different CV folds using all the training

data. This was dumb because | was just training the calibration model on fights the main model had
already been trained on leading to overfitting on train, and poor results on the holdout test dataset.

+ Expanded calibration set: Increased from 10% to 20% of data

* Parameter tuning: Adjusted y_min , y_max , and out_of_bounds settings

None of these approaches yielded meaningful improvements.

Platt Scaling: A Little Better

https://www.mma-ai.net/news 46179


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Platt scaling uses logistic regression to map uncalibrated probabilities to calibrated ones. While less
flexible than isotonic regression, it's much more suitable for small datasets.

class SimplePlattCalibrat ion:
def _init_(self, max_iter=100, random_state=42):
self.max_iter = max_iter
self.random_state = random_state
self.calibrator = None

def fit(self, y_prob, y_true):
from sklearn.linear_model import LogisticRegression

# Reshape probabilities for sklearn (needs 2D input)
y_prob_reshaped = y_prob.reshape(-1, 1)

self.calibrator = LogisticRegression(
max_iter=self.max_iter,
random_state=self.random_state,
solver='lbfgs'

)
self.calibrator. fit (y_prob_reshaped, y_true)

Implementation in Training Pipeline

The calibration was integrated into the training pipeline using scikit-learn's CalibratedClassifierCv :

# Three-way split for proper calibration validation
(Xtrain, y_train), (Xcal, y_cal), (Xtest, y_test) = split_data_three_way(
X, y, train_size=0.775, val_size=0.125

# Wrap AutoGluon predictor for sklearn compatibility
autogluon_wrapper = AutoGluonWrapper(predictor, feature_columns=X_train.columns.tolist())

# Create calibrated classifier using holdout method
calibrated_clf = CalibratedClassifierCv(

est imator=autogluon_wrapper,

method='sigmoid', # Platt scaling
prefit" # Use prefit since AutoGluon is already trained
ensemble=False  # Use single calibrator since we have proper split

# Fit calibrator on holdout calibration set
calibrated_clf.fit(x_cal_clean, y_cal)

Platt Scaling Results (Success!)

Calibration Results (sigmoid):

Should Use Calibration: True

Calibration Set Log Loss Improvement: 0.0098
Test Set Log Loss Improvement: 0.0107

Test Set Brier Score Improvement: 0.0043
Test Set ECE Improvement: 0.0174

Final Test Log Loss (Original): 0.5948
Final Test Log Loss (Calibrated): 0.5841

The improvement of 0.0107 in log-loss on the holdout test set of data represents a meaningful, if small,

gain in probability accuracy.

Profitability Impact Analysis

While the statistical improvements were modest, | wanted to examine whether calibration affected bettir

profitability. Here's the cumulative profit comparison between the original and calibrated models on

the

41179


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

test set:

Initial dankrell: $1000.00
Bee ancune" $10.00

| Finan punkrott | Total Profle | Total Magered | in % | ROT (&) | Sharpe Ba
ii piesa closing [$s iueas ss aaea Ts | yaret | as2ee |
[$e s ames | joiret | ialoex |
or within spet closing 1s ae | $ eet | $ ee | G2.oer | i
fuitnin-spet_sevencay 1S tases] ieecas |S deooson | silees | i
si-picked_ favorite closing 13 tases 1s 1$  aosoiee | iss | ao | Focenx | i
sipicked_ favorite sevenday |B ead /8 1s eo] tee | az | 75.90 | i
sipicked-positive 13 unis 1s 1s eo | 300 | ee | Ss.05x | i
si picked: positive 13 iasise 1 § 1s | te | Te | Sacaex | i
si-pickedindercoe closing 13 dusts 1s 1s | al el sell i
SiLpicked_underoog_sevencsy 13 tise | $ 1s [S| 35] Sa.zty | i
raposieive.w closing res ae 1s | as} be 5.50 | i
reer_posicivel_sevencay 13 jasies | § 1s | Zs! 98 | 30.30% | i
Instat it-opponene-spct edge closing | $ 6.90 | § 1s | ie} ze | as.zey | i
fnst al if apeanent fort eden sevenany 1S salt LE [del ns] Saw | t

Initial Gankroll- $1060.00
bee soune” $10.00

Sharpe ma

piesa closing 13 oi ae e.7ex | 3-608 |

sa plete teverdey 13 oo] se! ye. | 8.368 |
pieced evar within Spct closing 13 oo] is2 | dea | isan |
13 oo] isa | geez | is.sex |

13 oo] sa | ree | Gan |

favorites 13 oo] as | ream | een |
ched_positivenev_closing 1s eo] 357 | Gea | asa |
ched_positivenevasevenaay 1s al ems | tesa |
ched_aneerdog_cloving Is el a! Sam | bom |
ched_uncerdog_sevenday 1s ol] | Som| see |
ghee poritive tv closing i el San | Tea

The results show modest but consistent improvements across most betting strategies. The core strateg
(ai_all_picks_closing) improved from 13.26% to 13.60% ROI, while the seven-day advance strategy
increased from 14.98% to 15.30% ROI. Notably, win percentages remained identical at 70.70% which is
interesting because some picks do change based on the calibration process.

Interestingly, underdog-focused strategies saw slight decreases in ROI (35.31% to 34.97% for closing
odds) and slightly improved favorite win percentages (70.70% to 70.80%), suggesting the calibration
process may have made the model slightly more conservative on high-value underdog picks. Meanwhile
the poorly performing +£V on ANY fighter (regardless of Al pick) strategies remained unprofitable in bot
versions but dramatically improved in the calibrated version showing the calibration process working in
action. The Al-picked +EV strategy was basically the same though which tells me | need more profit
testing on where the +EV threshold is, like +5% win chance? +10% win chance? When do we bet agains
the Al pick? Like last event Kevin Holland was picked by Al at -180 or something yet Vegas had him at
-500. What's the threshold to pick against the Al? IDK yet but I'll go test it out. This is the most pressing
question when we're talking about betting strategy.

However, it's crucial to note that ROI is a moving target—betting markets evolve, line movement varies,
and small sample sizes can significantly impact results. These profit tests represent performance on a
specific test set and shouldn't be viewed as guaranteed future returns. The real value of calibration lies
more reliable probability estimates for bet sizing and strategy decisions rather than raw profit
maximization.

Model Performance Analysis

Here's how the calibrated model performed against Vegas odds across all unfiltered fights in the past 1!
years (meaning we include split decisions, and DQ's, and whatnot which aren't included in the model
training):

{
"yegas_odds_performance": {
curacy": 0.700,
"Log_loss": 0.563,
"brier_score": 0.194
1,
"mma_ai_performance": {
"accuracy": 0.710,
"Log_loss": 0.603,
"brier_score": 0.208
},

48/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
if

"mma_ai_performance_calibrated'
0.710,

0.598, # Improved
rier_score": 0.206 # Improved

The calibrated model maintains identical accuracy while providing more reliable probability estimates.

The Calibration Curve Analysis

The most telling evidence comes from the calibration curve itself. My uncalibrated model exhibited class
miscalibration patterns:

+ Underconfident in 50-60% range: When the model predicted 55%, fighters actually won ~60% o
the time

* Overconfident in 40-50% range: When the model predicted 45%, fighters won closer to 40% of
the time

def plot_calibration_curve(self, n_bins=10, include_all_fights: bool = False):
# Get calibrated predictions if calibrator is available
y_prob_calibrated = None
if self.calibrator is not None:
y_prob_calibrated = self.calibrator.predict_proba(test_data_clean)[:, 1]

# Calculate calibration curves
prob_true_model, prob_pred_model = calibration_curve(self.y_test, y_prob_model, n_bins=n_t
if y_prob_calibrated is not None:
prob_true_calibrated, prob_pred_calibrated = calibration_curve(
self.y_test, y_prob_calibrated, n_bins=n_bins

# Plot results
plt.plot([@, 1], [0, 1], 'k--', label='Perfect Calibration")
plt.plot (prob_pred_model, prob_true_model, 's-', label='Model (Original)')
if y_prob_calibrated is not None:
plt. plot (prob_pred_calibrated, prob_true_calibrated, '*-',
label='Model Calibrated’, alpha=0.8)

The Odds Inclusion Paradox

An interesting discovery was the relationship between including betting odds and calibration. In previous
experiments 1-2 years ago, calibration consistently hurt performance. The key difference was | was
including Vegas odds as features.

With Odds (Historical)

* Pros: Exceptional calibration, higher accuracy (~73-74%)

* Cons: Model HEAVILY favored odds over other features, struggled with underdog picks, -6% lower
ROI against all model picks

Without Odds (Current)

* Pros: Better underdog detection, higher betting ROI, room for calibration improvement

* Cons: Lower accuracy (~71%), requires manual calibration

This represents a fascinating tradeoff: accuracy vs. profitability. The model without odds hits
significantly more profitable underdog picks, even though its raw accuracy is lower.

49/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Why Calibration Matters for Bettors

While calibration may not be necessary for a profitable model (mine was already profitable before
calibration), it provides crucial benefits for bet sizing and strategy:

Kelly Criterion Application

With well-calibrated probabilities, bettors can use the Kelly Criterion more effectively:

def kelly_bet_size(prob, odds, bankroll):
“Calculate optimal bet size using Kelly Criterion™""
decimal_odds = american_to_decimal(odds)
edge = prob * decimal_odds - 1

if edge <= 0:
return 0

kelly_fraction = edge / (decimal_odds - 1)

return min(kelly_fraction * bankroll, bankroll * 0.1) # Cap at 10%

| experimented with fractional kelly in the past. It's promising long term, especially since the model's
logloss is consistently less than .059, but the wild swings are too painful. If we got the model to be lowe!
logloss than Vegas by including the odds in the feature set | think this is promising but will experiment

more later with that.

Calibrated probabilities enable more sophisticated betting strategies:

def edge_based_betting_strategy(predictions, min_edge=0.05):
“select bets based on edge over Vegas odds rather than absolute confidence."
betting_opportunities = []

for fight in predictions:
# Get model probability and Vegas implied probability
model_prob = fight['model_confidence']
vegas_decimal_odds = american_to_decimal(fight['vegas_odds'])
vegas_implied_prob = 1 / vegas_decimal_odds

# Calculate edge (model probability - market probability)
edge = model_prob - vegas_implied_prob

# Only bet if we have a significant edge
if edge >= min_edge:
kelly_fraction = edge / (vegas_decimal_odds - 1) # Kelly criterion
bett ing_opportunities.append({
‘pick': fight ['fighter_name'l,
"edge': edge,

oa)

return betting_opportunities
I'll probably implement something like this later on after | do more profitability backtesting with flat units

Technical Implementation Details

The final calibration system integrates seamlessly with the existing prediction pipeline:

def _get_model_predictions(self, test_data, use_calibrated=None):
"Get model predictions, optionally using calibrator."""
if use_calibrated is None:

use_calibrated = self.calibrator is not None

# Get original predictions

50/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
y_pred = self.predictor.predict(test_data)
y_prob = self.predictor.predict_proba(test_data)

# Apply calibration if available
if use_calibrated and self.calibrator is not None:
test_data_clean = test_data.drop(columns=['sample_weight'], errors='ignore')
y_prob = self.calibrator.predict_proba(test_data_clean) [:, 1]
y_pred = (y_prob > 0.5).astype(int)

return y pred, y_prob

Lessons Learned

1, Dataset size matters: Isotonic regression requires substantial data; Platt scaling works better
with limited samples

2. Validation strategy is crucial: Proper train/calibration/test splits prevent overfitting
3. Calibration # Accuracy: Better probabilities don't always mean better classifications

4. Feature engineering impacts calibration: Including odds improves calibration but hurts
profitability

5. Domain expertise guides tradeoffs: Understanding the betting market informed the decision 1
exclude odds

Conclusion

While the final improvement in logloss was modest (0.0107), it represents a meaningful step toward mor
reliable probability estimates.

The key insight is that calibration serves different purposes depending on your goals. For pure
classification accuracy, it is unnecessary. For models that are already well calibrated (like when you
include the odds, or use algos that are better for calibration like Random Forest) calibrating the
predictions using Platt or Isotonic harms the output.

The surprising relationship between feature inclusion (odds) and calibration highlights the complex
tradeoffs in machine learning systems. Sometimes the most accurate model isn't the most profitable on
and the most calibrated model isn't the most accurate one.

For practitioners working with limited datasets, Platt scaling offers a robust path to improved calibration

The simplicity of logistic regression makes it both interpretable and reliable, even when more
sophisticated methods fail.

51/79


1/6/26, 9:08 AM MMA-ALnet

Machine Learning for Sports Prediction:
Should You Balance the Winrate of Competito
1vs Competitor 2?

June 19, 2025

Notat this time. I've spent so many hours investigating this in my free time. ROI is our north star, with
logloss and accuracy being secondary metrics. This isn't a perfect test because the betting markets shi
as time moves. For example, 2016 saw Vegas favorites win only 61% of the time. 2024 saw favorites win
70% of the time and UFC red corner is usually the betting odds favorite. So this isn't the end-all be-all
conclusion, but it is absolutely the point in time conclusion

The long and short of the massive amount of trial and error is this backtested ROI based on the last year
of fights the model has never seen:

ROI Comparison: Balanced vs Unbalanced Training

ROl of balanced 50/50 fighter1 win/lose:

RO1 of unbalanced 59/41 fighter1 (UFC assigned red corner) win/lose:

https://www.mma-ai.net/news 52/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Based on how | do feature engineering and model tuning, balancing the fighters' win rate before training
the model is terrible. If | do some various feature and model tuning | can see the balanced model start to
perform closer to the unbalanced model but on average it is basically always lower ROI. | have
interrogated my code a thousand times with Claudea Opus and Gemini 2.5 Pro to try to suss out any
logical errors and | cannot find any.

Why Does Balanced Training Destroy Performance?

“Begin speculation*

Distribution shift between training and inference creates systematic probability miscalibration.
When you balance the dataset to 50/50, you're teaching the model that P (fighter1_wins) = 0.5 across all
feature combinations. But in reality, fighter1 (red corner) wins ~60% of the time because the UFC
systematically assigns the red corner to champions in title fights and generally more experienced/favore

fighters in regular bouts.

The calibration mismatch:

* Training: Model learns P(fighter1_wins | features) where the base rate is artificially 0.5

* Inference: Model predicts on data where the true base rate is 0.6

This specifically destroys betting ROI because:

Systematic underestimation of favorites: When a red corner fighter should win with 70%
probability, your balanced model might predict 60%, causing you to miss profitable favorite bets

Systematic overestimation of underdogs: When a blue corner fighter should win with 30%
probability, your model might predict 40%, leading to negative EV underdog bets

Market inefficiency amplification: The higher RO! without balancing implies that the model is
learning this pattern of corner bias more efficiently than the market is.

The model's probability outputs are fundamentally miscalibrated for the real world's corner assignment
bias. You're essentially training a model for a balanced fantasy UFC and then applying it to the
systematically biased real UFC, where corner assignments carry predictive information about fight
outcomes.

Bottom line: The balanced training throws away valuable signal (red corner = usually stronger fighter) a
teaches the model incorrect base rates, leading to systematically poor probability estimates that destroy
betting performance

Doesn't Calibrating the Winrate Just Create a Proxy for the Odds?

53/79


1/6/26, 9:08 AM MMA-ALnet

No, and here's why that concern misses the point:

The model isn't learning "red corner = bet favorite." It's learning complex feature interactions from
granular performance statistics that happen to correlate with corner assignments. The red corner
correlation exists because the UFC assigns corners based on ranking, championship status, and
experience - the same underlying factors that drive fight outcomes.

Key distinctions:

* Betting odds reflect public perception, line movement, and bookmaker risk management

+ The model analyzes actual performance metrics: strike accuracy trends, takedown defense pattern
cardio indicators, opponent-adjusted statistics, etc.

+ The value comes from divergence identification based on as much statistical information as possibl
The corner bias is simply real world information that the model can learn to incorporate better than
average bettors.

The unbalanced training preserves the signal that corner assignments carry meaningful information abo
fighter quality, information that's already baked into the real-world problem you're trying to solve.
Throwing away that signal artificially handicaps the model's ability to make properly calibrated
predictions.

What About Including the Odds?

This is a hotly debated topic in the sports betting community. Bill Benter, one of the fathers of algo
betting, argued that the odds are best to be included because they encode so much information. Having
been testing this hypothesis for many thousands of hours, my conclusion is that if you don't have a certi
level of extremely high quality engineered features, then yes you should include the odds. But at a certa
point, the features you engineers will actually encode more information in combination than the odds do
At that point, including the odds simply lowers the ROI.

Mathematical Mechanism

When you include odds as a feature, you're introducing a variable that represents the market's
aggregated probability estimate: P_market(fighter1_wins). This creates several mathematical problems:

1, Feature Dominance and Prediction Convergence
Tree-based models will heavily weight the odds feature because it exhibits high mutual information with
the target across the entire dataset. The model's predicted probabilities become:

P_model(fighter1_wins) = aP_market + (1-a)-P_stats where a > 0.5

This forces convergence toward market consensus.

2. Outcome Distribution Skew

Including odds biases the model toward predicting high-frequency, low-profit outcomes (favorites). You
engineered statistics without odds bias toward identifying low-frequency, high-profit outcomes (underd
value).

3. +EV Prediction Accuracy Inversion
The critical insight: RO! optimization requires accuracy specifically on profitable bets, not overall
accuracy.

* With odds included: Model achieves ~74% accuracy but concentrates correct predictions on
favorites (odds = 1.2-18, profit margin = 20-80%)

* Without odds included: Model achieves ~71% accuracy but concentrates correct predictions on
underdogs (odds = 2.5-4.0, profit margin = 150-300%)

https://www.mma-ai.net/news 54/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
ROI Mathematics

Consider two hypothetical models:

* Model A: 75% accuracy, predicts favorites 85% of the time > Expected ROI = -2% (high accuracy,
low margins, vig erosion)

+ Model B: 50% accuracy, predicts underdogs 60% of the time > Expected ROI = +15% (lower
accuracy, exponentially higher margins)

The fundamental equation:

ROI = E(accuracy_i x frequency_i x profit_margin_i)

Your empirical results demonstrate that excluding odds increases accuracy_underdog dramatically while
slightly decreasing accuracy_overall. Since profit_margin_underdog >> profit_margin_favorite, the ROI
optimization occurs through maximizing performance on the subset of predictions with the highest prof
potential.

Information Compression Loss

Odds represent market consensus based on public information flow. Your engineered features capture
orthogonal signals that specifically identify cases where statistical analysis diverges from public
perception—exactly the scenarios that generate +EV opportunities. Including odds suppresses these
divergent signals in favor of consensus alignment, destroying the edge that creates profitable betting
opportunities.

The Vanity Metrics Problem in Machine Learning

You will see many other novice machine learning engineers practice their skills against sports prediction
They will calibrate their models against vanity metrics like accuracy in most cases. They will see
evaluations like 85% accuracy, then not have any motivation to figure out why their model will fail in the
real world then make a YouTube or Medium post about it.

This represents the fundamental disconnect between academic machine learning and profitable real-
world application. The sports prediction space is littered with impressive-sounding accuracy claims that
evaporate when subjected to actual betting markets. These engineers optimize for metrics that sound
impressive in blog posts rather than metrics that generate alpha.

The vanity metrics obsession creates a particularly insidious blind spot: data leakage and overfitting
become invisible when you're chasing high accuracy numbers. Consider the classic example in YouTube
tutorials (https://wwwyoutube.com/watch?v=LkJpNLlaeVk) where a model achieved impressive accurac
predicting UFC fights using Elo ratings... except they used post-fight Elo ratings for training, meaning th
model literally saw fight outcomes during training. This is textbook data leakage: using future informatio
to predict past events.

Data Leakage Examples in Sports Prediction:

Training on post-game statistics to predict game outcomes

Including betting line movements that occurred after the event

Using season-end rankings to predict mid-season games

Including opponent-adjusted metrics calculated after the fight
The overfitting trap compounds this problem. Hyperparameters are the configuration settings that contr

how a model learns: things like learning rate, tree depth, regularization strength. | can easily tune these
hyperparameters to achieve 78% accuracy on my validation set by optimizing for that specific data slice

55/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
But this creates a model that memorizes the validation set's quirks rather than learning generalizable
patterns.

The feedback loop is broken: High accuracy on test data > immediate gratification > publish results -
never discover real-world failure. There's no motivation to dig deeper because the vanity metric has bee
satisfied. The engineer never learns that their 85% accuracy model would lose money consistently
because they never actually test it against betting markets.

Real-World Adversarial Markets

Real-world sports betting is an adversarial market where you're competing against:

Professional odds compilers with decades of experience

Sophisticated betting syndicates with proprietary data

Market makers who adjust lines in real-time based on money flow

The inherent vig that makes break-even betting a losing proposition

Simply achieving high accuracy on historical data means nothing if your predictions can't consistently
identify mispriced markets. The difference between academic exercise and profitable application is the
difference between predicting outcomes and finding edges.

Lessons from Four Years of Iteration

I'm four years into this space and I've made all the same mistakes. I've built models that achieved 78%
accuracy and lost money consistently. I've spent months optimizing for log loss improvements that
translated to worse ROI. I've fallen into every trap outlined above because the feedback loop between
model performance and real-world profitability is opaque until you actually start tracking betting results
over extended periods.

This is a far more complex problem than | ever initially thought 4 years ago, but I believe what's outlined
above is one of the reasons I've been seeing almost 20% ROI over the past few years in my free, public
predictions. The key insights around dataset balancing, odds inclusion, and ROI-focused optimization
came from years of iterative failure and debugging, not from following standard ML tutorials.

The engineering rigor required to build profitable models demands treating accuracy as a vanity metric
and ROI as the only metric that matters with the knowledge that betting markets shift as time goes on at
you must constantly retrain the model to match the current zeitgeist. Hence why I'm on version 6.3 right
now. I've redone 100% of the code 6 times now. If you want to avoid the same mistakes I've made, feel
free to reach out. I'm an open book on exactly how | built this model, no secrets. And also, as usual,
shoutout to Chris from Wolftickets.ai for being one of the very very few people in this ML for sports
prediction field that actually shared his technical knowledge and saved me endless hours of wasting my
time. | like paying that forward to others who are interested in this space.

56/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Why Don't You Recommend The Positive EV
Picks?

April 19, 2025

I have released v6.1 of the model. Large improvement in accuracy but most specifically in ROI. So let's ti
betting strategy. One question | get asked practically every day is, "why do you recommend only the Al
picked winners even if they're not +EV?"

The best tech available for predicting UFC are binary classification algorithms. These are algorithms
specialized in classifying a fighter as either 1 or 0, win or loss. The confidence score that the algorithm
comes up with is not its fundamental strength.

"So why don't you calibrate it?"

Because the calibration tech we have for these algorithms suck. Platt Scaling and Isotonic Regression at
the two most potent weapons we have to calibrate these predictions. | have experimented a lot with ther
100% of the time they suck. They lower the accuracy and the ROI. I've experimented with custom
calibrations but it's just too inconsistent and | don't see a promising way forward. | would absolutely love
to have better calibration, so if you have any ideas please let me know. That being said, the ultimate goa
here is risk-adjusted returns. Calibration is a tool to help us get there but it's not necessary to acheiving
the goal.

Anyway, who cares about calibration if we're making fat ROI? Take off your sports bettor hat and all the

"fundamentals" you know like +EV is all that matters to be successful. Put on your machine learning cap
and look at the data. This is the calibration curve:

Calibration Curve

104 --+ Perfect Calibration
Model (Test)
= Vegas (Test)

o8

06

oa

Fraction of Positives

02

oo 02 oa os oa 10
Mean Predicted Probability

Note how the model is highly underconfident in its 50-65% confidence score picks and highly
overconfident in it's 35-50% confidence scores. The nature of the algorithm is to maximize it's ability to
successfully pick the winner and it does this at the expense of accurate calibration. It clusters it's
confidence scores in the 50-65% range. AND THAT'S OK!

Here's the model evaluations and vegas evaluations on the last year of unseen fights:

57/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

‘al Performance Metrics Comparison

Metric Vegas Odds MMA Al Model
Accuracy 0.690 0.710
Precision 0.725 0.697
Recall 0.725 0.859
FI Score 0.725 0.770
Log Loss 0.587 0.602
Brier Score 0.201 0.207

Note the model is more accurate than Vegas, but it's logloss and Brier scores (essentially real world win
are worse. Again, THIS IS OK. Why? Because look at these profit strategy backtests over the last year of
unseen fights:

Backtest ROI by Strategy

parlay_3_legs_al picked postive ev closing
parlay_3 legs_al picked positive_ev_sevenday
parlay 2 legs_ai picked positive ev closing
parlay_2 legs_al_picked postive ev sevenday
3 legs_alrancom_picks sevenday
al_random.picks closing

ai picked underdog sevenday
al_peked_underdag closing
parlay_2_legs ai random picks closing
parlay_2 legs ai random picks sevenday
{top_confidence.sevenday
legs_al_top_confidence closing
parlay_2 legs +
21 picked_positive ev closing

21 peked_positive ev sevenday

parlay_2 legs al top confidence closing

a. picked_ev_or within Spet_closing

al picked | Spet_sevenday

ai al_picks sevenday

al all picks closing

parlay_2 legs any fighter positive ev closing
21 picked favorite sevenday
al picked favorite closing
opponent Spet_edge_closing
‘opponent spct_edge.sevenday
any_fighter_postive_ev_closing
any fighter positive ev sevenday
parlay_2legs_any fighter positive ev sevenday

top_confidence sevenday

strategy

st

bet_against,
bet agains

os > a 30 75 100 us 350
Return on Investment RO} (%)

The fundamental strategy of betting on 100% of the Al picks on the closing odds is 14% profitable
(ai_all_picks_closing). The strategy of picking only the +EV fighter based on the Al confidence score
whether the Al picked that fighter to win or not is -0.5% unprofitable (any_fighter_positive_ev_closing).

Interpreting The Strategies

None of this is to say that +£V isn't a helpful metric. Look at another fundamental strategy, betting only
+EV Al-picked fighters (ai_picked_positive_ev_closing): +24.2% ROI. 10% higher than just doing down t
line bets but it means sometimes there'll be events where you don't get to bet which makes me sad.

So let's put this all together. Why are parlays so good with this machine learning algorithm? It's because
the algorithm is so gosh darn good at picking winners. Parlaying the Al picks, whether +EV or not, is a
highly profitable strategy. The best strategy is doing 2 to 3 leg parlays on the +EV Al picked fighters. I'll
probably move into posting these parlays again in the homepage.

58/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

EDIT: Shoutout to @Heisenb3rg on X for bringing Sharpe Ratios to my attention! This is a crucial metric
from finance that measures risk-adjusted returns. Essentially, it tells us how much return we're getting fc
the amount of risk (volatility) we're taking on with a particular betting strategy. A higher Sharpe Ratio is
better.

Backtest Sharpe Ratio by Strategy

ai pete underdog_sevenday TT 0:7

Lal peks_sevenday ows

fates, closing 038

ai pled postive ev closing 038

2 picked ndedog closing —_——?a5 37
parlay_3_legs_ai_random_picks_closing a 036
parlay_2Iegs_fandom pcs, losing oo a36
pariny 2 legs rondom picks seventy a 036

2: pcked ev oF within spt sevenday i 3s
parlay_3_legs_ai_picked_positive_ev_closing i 0.35
patio. 2 legs. picked postive ev sevenday a3s

spare. 2 fps pled postive. ev losing i 3s
3 ‘a. picked ev_or within Spet closing i 033
8 parlay_2_legs_ai_top_confidence sevenday 033
orty_3 legs random picks sevendoy ast
para 3 les.a picked postive ev sevenday i 28
parlay_3lepsal top condence sevenday i 026
allay? tegs tp, confidence losing a26
picted favorite sevenday a3

fi pled foreclosing a ozs

patty 3 Jogs fp, confidence. losing oo at
‘bet_against_ai_if opponent_spct_edge closing sd 0.10
bet agains opponent, spc edge. sevenday = 0s
arlay_2Iegs_ay. fighter postive, ev-clesing = aoe
ty fighter postive, ev_eosing on.

any tighter postive, ev sevenday 08

party 2 legsany fighter postive. ev sevenday oar

03 om os:
‘Sharpe Ratio (Risk-Acjusted Return)

Looking at the Sharpe Ratio analysis, we can see how different strategies compare not just on raw ROI,
but also on how bumpy the ride is. For v6.1, the 'ai_picked_positive_ev_closing' strategy seems to offer
strong balance, providing good returns without excessive risk compared to some others. However, the
‘ai_all_picks_closing' still performs well from a risk-adjusted perspective, offering broader betting
opportunities.

So, what's the takeaway? While different model versions might favor underdogs or favorites differently,
the data consistently points towards a simple, robust approach. Considering both the ROI and the Sharp
Ratio (risk-adjusted return), the strategy of betting on all Al picks using the odds available 7 days
before the event (ai_all_picks_7day) emerges as the winner. It offers the best blend of profitability and
lower volatility, making it the simplest and most effective core strategy based on the current analysis.

It's important to remember that model performance, especially regarding favorites versus underdogs, ci
shift between versions and timelines. Therefore, relying solely on specific odds ranges like underdogs
only might be less reliable long-term despite being very profitable over the past year. Instead, | think
focusing on fundamental strategies evaluated through metrics like ROI and Sharpe Ratio is key. For the
current model, based on the patterns I've seen over the years, | think I'm going to stick with the core
strategy ai_all_picks_sevenday which provides the best risk-adjusted performance indicated by the
Sharpe Ratio. To get higher returns at higher risk, 2 to 3 leg parlays of your choice are a good option.

59/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

MMA-AI UFC 314 Predictions

April 12, 2025

Here are the Al predictions for UFC 314.

Al Predictions - UFC 314

60/79


1/6/26, 9:08 AM MMA-ALnet

Al Predictions and Analysis for UFC: Emmett
vs Murphy

April 4, 2025

Check out the latest video analysis and Al predictions for the upcoming UFC matchup between Emmett
and Murphy. The Al generated interpretation of my face always cracks me up.

Al Predictions for UFC Emmett vs Murphy - MMA-Al.net

https://www.mma-ai.net/news 61/79


1/6/26, 9:08 AM MMA-ALnet

New Video: Understanding MMA-AI
Predictions

March 19, 2025

Check out our latest video explaining how MMA-Al makes its predictions and why our model has been
consistently outperforming Vegas odds.

UFC Fight Night: Moreno vs Erceg - Al mathematical predictions

https://www.mma-ai.net/news 62/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Demystifying the MMA-AI Prediction
Algorithm: How We Predict UFC Fights

March 18, 2025

After countless requests from the community, I'm finally pulling back the curtain on how our UFC fight
prediction algorithm works. This isn't just another black-box system—it's a carefully engineered, multi-
level approach to fight analysis that's consistently outperforming Vegas odds.

Why Share Our Secret Sauce?

Many have asked why I'd reveal the inner workings of a profitable system. The truth is simple: I don't
believe in keeping this knowledge secret. Much of this algorithm was developed with the assistance of A
tools that generated about 90% of my code. As someone who's been an open source developer for 15
years, | firmly believe that open knowledge sharing accelerates human progress. | don't have a moat to
protect—and frankly, the more people who understand these concepts, the faster our collective
knowledge will advance. Sports prediction is hard enough that even with this blueprint, you'll still need t
invest thousands of hours to truly master it.

The Four Levels of Fight Prediction
Level 1: Foundation Stats + Bayesian Gamma-Poisson Smoothing

We start with raw fight statistics—strikes thrown, takedowns landed, control time, and dozens more. But
raw numbers can be misleading, especially for rare events like submissions. This is where Bayesian
Gamma-Poisson smoothing comes in.

Imagine a fighter who's attempted only one submission in their career and landed it. Is their submission
accuracy really 100%? Probably not. Gamma-Poisson smoothing helps us balance observed data with
prior knowledge, preventing outliers from skewing predictions. For high-volume stats like significant
strikes, the smoothing effect is minimal. But for rare events like submissions or knockdowns, it provides
crucial stability by "pulling" extreme values toward more realistic expectations based on the fighter's
division and overall UFC averages.

Level 2: Comparative Analysis

Next, we calculate a fighter's efficiency metrics: accuracy (how often attacks land), defense (how often
they avoid opponents’ attacks), output per minute, and the ratios between fighters across all statistics.
This gives us a clearer picture of how fighters perform relative to each other, beyond just counting
actions. A fighter might land fewer strikes overall but have significantly higher accuracy—a crucial
distinction our model captures.

Level 3: Time-Weighted Averages + Variability

MMA evolves rapidly, and a fighter today isn't the same as they were three years ago. We calculate both
standard and time-decayed averages with a 1.5-year half-life for all statistics. This means a fight from
three months ago has far more impact on our predictions than one from three years ago.

Additionally, we measure the variability of these stats using Median Absolute Deviation (MAD) instead of
standard deviation. MAD is less affected by extreme outliers, providing a more stable measure of how

consistent a fighter's performance has been.

Level 4: Adjusted Performance (AdjPerf)

63/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet
Our most sophisticated metric answers a critical question: how does a fighter perform against a specific

opponent compared to how that opponent's previous adversaries performed? This z-score normalizatio
looks like:

stat_adjperf = (fighter1_stat - fighter2_stat_prev_opp_avg) / fighter2_stat_prev_opp_mad

In plain English: we're measuring how much better or worse a fighter performed against their opponent
compared to what we'd expect based on the opponent's history. If a fighter lands more strikes against
someone who's historically difficult to hit, that's far more impressive than landing the same number
against someone who's easily hit. AdjPerf captures this crucial context.

Bu

ig the Predi in Model

With thousands of engineered features from these four levels, we use Autogluon (thanks to Chris from
Wolftickets.ai for this recommendation) to train our prediction model. Using presets like "experimental"
and time-ordered data splitting (typically 80/20 or 90/10 train/test), we run extensive cross-validation to
ensure reliability.

There's no definitive guide to sports prediction—we've conducted thousands of hours of testing to find
what consistently beats Vegas odds. Everything from feature selection to hyperparameter tuning require
constant experimentation and refinement. What works for NFL might not work for UFC, and what workec
last year might not work next year.

The final model combines about the 30 best of these features, weighted according to their predictive

power. The result is a system that consistently identifies value bets where our predicted win probability
exceeds what Vegas odds imply.

The Practical Limitations of ML Probabi

Ss

One of the humbling lessons I've learned over the years is about the relationship between machine
learning and betting strategy. Despite the sophisticated nature of our model, I've had to accept that ML
algorithms have inherent limitations when it comes to probability calibration. While traditional betting
approaches rely heavily on expected value (EV) calculations, I've discovered that applying these same
principles directly to ML outputs can be problematic. Our tabular ML models excel at binary classificatio
—essentially determining which fighter is more likely to win—but their confidence scores aren't
necessarily true probabilities in the statistical sense. Even when optimizing for log loss (which
theoretically improves probability calibration), there remain subtle biases and distortions in how these
models estimate probabilities. Through trial and error, I've found that treating the model's outputs as
relative confidence levels rather than exact win probabilities leads to more consistent results. Instead of
rigidly applying EV formulas, we use confidence thresholds as a filtering mechanism to identify promisin
bets. This pragmatic approach acknowledges the model's strengths in pattern recognition while
respecting its limitations in precise probability estimation—a balance that has proven more effective that
assuming perfect calibration.

64/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Sports Betting Odds vs. ML Probability Distributions

Hard to find EV bets when ML probabitieg
MIL rarely predicts extreme ations
probabilities

Frequency

Sports Betting Odd

52° ML predict_proba()

0% 20% 40% 60% 80% 100%
Win Probability (%)

‘ML models tend to be conservative in predictions due to calibration objectives, regularization, and limited features

Visualization of our model's prediction distribution compared to actual outcomes, showing the effectiveness of o
approach despite calibration challenges.

Why Current LLM-Based Fight Predictions Fall Short

I've extensively tested Large Language Models (LLMs) like GPT-4 for fight predictions, and the results
have been consistently disappointing. The fundamental limitation is clear: today's LLMs lack the ability t
analyze fight footage. They're trained primarily on text data, which means they miss crucial visual
information—a fighter's movement patterns, subtle defensive vulnerabilities, changes in stance, or signs
of fatigue that only appear on video. Statistical data can tell us a lot, but the visual dimension of fighting
contains irreplaceable insights that no spreadsheet can capture. The good news? This limitation is
temporary. Within the next 1-2 years, we'll see multimodal Al systems trained on vast libraries of video
footage, capable of analyzing thousands of fights frame-by-frame. When that happens, Al fight predicti
will undergo a revolutionary leap forward. Until then, the most effective approach remains combining
sophisticated statistical modeling with human expertise for context and interpretation.

Conclusion: The Never-Ending Journey

MMA prediction remains as much art as science. While our technical approach provides an edge, the
sport constantly evolves, and so must our model. Every event brings new data, every fighter brings new
patterns, and our system continually adapts to these changes.

Whether you're a casual fan or a serious bettor, | hope this behind-the-scenes look helps you appreciate

the depth of analysis that goes into each prediction you see on MMA-Al.net. And if you're inspired to bu
your own model? Even better—innovation thrives when knowledge is shared.

65/79


1/6/26, 9:08 AM MMA-ALnet

Model v6 Release

March 15, 2025

Model v6 is available for tonight's event and events going forward. This is intended to be a long-term
release.

Changes:
* Unit tests for everything

© Finally we have unit tests for all calculations so | can sleep at night. At least now if the model fai
| know it's not some unknown bug.

+ Bayesian Gamma-Poisson

© We switched out from just doing Bayesian Beta Binomial smoothing in accuracy/def and per_mi
calculations to a Bayesian Gamma-Poisson for all stats starting at the very beginning. This is a
more robust and consistent smoothing factor.

+ Swapped stddev for median absolute deviation for the zscore normalization stat adjperf

© Adjperf layer of feature engineering changed from:
stat_adjperf = (fighter1_stat - fighter2_stat_prev_opponents_avg) /
fighter2_stat_prev_opponents_stddev
to:
stat_adjperf = (fighter1_stat - fighter2_stat_prev_opponents_avg) /

fighter2_stat_prev_opponents_mad

© Median absolute deviation (MAD) is a more robust measure of deviation for sports statistics
because it's not as vulnerable to outliers.

Future:

Have a few stats I'd still ike to engineer especially around the win conditions of submissions vs KO vs
decision.

I have also added visualizations to the predictions. Just hover over the fight or click on the fight to see
how each fighter compares to each other using a normalized stat difference. Note that these
visualizations do not account for how the model is weighing each stats, it's just the normalized differenc
in the stats between the fighters to give you an idea of where each fighter has an advantage. Larger
surface area = better stats but you may notice sometimes the fighter with the larger surface area is not
picked, this is due to the fact the stats are not weighed equally by the model and also because the mod:
makes some complex relationships between the stats that won't be reflected by the visualization.

https://www.mma-ai.net/news 66/79


1/6/26, 9:08 AM MMA-ALnet

Unit Testing Not Done Yet

March 1, 2025
Based on the past 2 months of performance I'm fairly convinced I have some bugs in the nonunit tested
code. I've started the unit testing but I'm not done yet so take this event (3-1-2025 UFC Fight Night) wit

a grain of salt, might be a good one to skip for now, but | should have lots of time after work to finish uni
testing next week and I'll post an update about bugs found once I'm done.

https://www.mma-ai.net/news 67/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

v5.2 Model Release

February 22, 2025

After countless hours of trial and error testing this week, we're implementing a few key refinements to tr
model. Rather than making sweeping changes, we're focusing on targeted improvements that have shov
consistent benefits in our testing:

1. Updated Training Data Window

We're moving the cutoff date for training data from 2014 to 2016. While I've been using 2014 as the
starting point for the past 4 years, it's time to shift forward to better capture recent trends in the sport.
MMA evolves rapidly, and data from 10 years ago may not be as relevant to today's fighting landscape a
it once was.

2. New Feature: UFC Age

We've added a simple but effective new stat: UFCAge. This measures the amount of time a fighter has
been in the UFC. While straightforward, this metric helps capture valuable information about a fighter's
experience at the highest level of competition.

3. Enhanced Feature Layering

Previously, we were using two primary layers of features:

* Decayed adjusted performance (dec_adjperf_dec_avg)

* Opponents' decayed average (opp_dec_avg)

Now we're adding a third layer: the fighter's individual stats (like strikes_landed_ratio_dec_avg). This
addition has proven valuable - by combining all three layers of a key base stat (like head_landed_ratio),
the algorithm seems better equipped to predict how each fighter will perform relative to the broader UF‘
population.

Looking Forward

I'm preparing to start producing analysis videos to help explain the model's decision-making process an
break down specific fight predictions. The site has been updated with the latest calibration curve, mode
evaluations, and feature importance rankings to reflect these changes.

As always, these updates are focused on one goal: improving our ability to predict fight outcomes

accurately. The changes might seem subtle, but in the world of fight prediction, small edges often make
the difference.

68/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Adjusting for Adjusted Performance and
Sneaky Bug Fixes

February 8, 2025

Goddamn that took forever. There was a bug in the zscore normalization stat | named _adjperf. If you
recall, the calculation for adjperf is:

stat_adjperf = (fighter1_stat - fighter2_prev_fight_stat_opp_avg) / fighter2_prev_fight_stat_

* £2_stat_opp_avg is the historical opponent average against fighter2 pulled from fighter2's previous
fight.

* £2_stat_opp_sdev is the historical opponent standard deviation pulled from fighter2's previous fight

This stat is extremely complicated to implement and was the reason we were predicting so few fights ev
since v5 was released with it included. We train with only data known pre-fight so we shift all the data
backwards by 1 fight for all fighters. The consequences:

Case #1:

All fights where at least one fighter is debuting are immediately eliminated because their data is 100% n
after we shift the data backwards by 1 fight and we can make no comparisons between them and their
opponent.

Case #2:

All fights where at least one fighter has only 1 previous fight is useless because one-previous-fight fight
has a 0 in fighter2_prev_fight_stat_opp_sdev for their opponent's adjperf calculation.

Case #3

One fighter has 3 previous fights, the other fighter has 2. Again, adjusted_performance for fighter1 will k
0.

And so on. It means that _adjperf is causing an enormous amount of fights to be dropped.

I've tried to fix this by calculating an initial standard deviation for all first time fighters. | collect all stats 0
first-time fighters on a per weightclass, per stat basis. So your _adjperf score against first-time fighters
based on how good or bad you did compared to the median first time fighter's stats. This feels fine,
because we're dropping all fights where either fighter only has 2 previous fights anyway.

Bug Fixes

So there was a fun little bug affecting ~2% of the training data where fighter2_stat_opp_sdev would enc
up being 0.00000002 or something so the adjperf score would end up being like 11,839,902 instead of «
more normal 1.5 or ~1 (ie., fighter! scored 1.5 standard deviations above what fighter2 usually allows in
that stat from their historical opponents). The reason was because | only use time-decayed averages in
the final model to capture recent trends in both fighters and their opponents and the time-decayed
standard deviation calculation is a bit complex:

weight = EXP(-A * ((T - t) / 365.25))

69/79


1/6/26, 9:08 AM MMA-ALnet
Sooooooometimes, T = 7.777777777 and t = 77.77777778 due to rounding in Postgres double precision.
have fixed this by capping the minimum standard deviation of stats to their 5th percentile. This is to avoi
insane standard deviations in tiny edge cases like Khamzat Chimaev's early UFC run where his opponen
only scored like 4 strikes against him in 4 fights then he fought Gilbert Burns who hit him 223 times. Thi:
meant Burns’ _adjperf score was insane because he took Khamzat like 400 standard deviations away
from his previous opponent averages.

Last, |had some small bugs affecting the filtering of training data. The order of filtering the data matters
LOT and | was doing some dumb stuff like filtering out Draws and stuff, THEN calculating number of figh
for each fighter and filtering on number of fights which is wrong because | already filtered out fights
before calculating number of fights for each fighter.

OK! That about wraps up the updates this week. It took a really long time to get through all those bugs s
I'm trying to get the updated predictions out before this event, bear with me.

https://www.mma-ai.net/news 70/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Updates to Betting Strategy

January 18, 2025

First, we're changing the wording from "Al win%'" to "Al confidence". This is to sort out the confusion the
many traditional bettors have. This algo is not specialized in finding +EV. It is a binary classification
algorithm, meaning it is very good at figuring out who will win, but not the real world win % chance.
Predictor.predict(upcoming_fights) returns a binary value of 1 = fighter1 predicted to win, 0 = fighter1
predicted to lose. We can do predictor.predict_proba(upcoming_fights) and get it's best guess at each
fighter's implied probability but generally this probability is not well-calibrated because this is specifical
a binary classifier.

Second, the reason we're doing parlays is because they're essentially a multiplier of the profit from the
single picks. You can risk less units to win more. You don't have to do this, you can just do the straight
picks and limit the boom and bust cycle of the parlay strategy, but backtesting is showing the parlay
strategy consistently outperforms the straight picks in ROI.

On this note, up until now I've been randomly creating the parlays. I've been trying to figure out how to ¢
rid of that randomization. Sorting the fights by Al confidence or inconfidence ended up performing wors
than random picks. So | thought about what does the algorithm crave to become more accurate? Data.

The new parlay strategy is based on backtesting showing more consistent results by sorting the parlays
by total combined fights both fighters have had. Islam has had 16 fights in the UFC. Renato Moicano has
had 17 fights. That's 32 fights of data that the algorithm has had time to calibrate on. So whatever the
algorithm picks on this fight is likely to have the most consistent accuracy. That's why we're seeing islan
in 3 parlays this event.

This strategy could use more refinement, and ideally we'd actually just create every possible 3 leg parla
and set that up as the predictions for every event. The only reason | don't do that is because it takes
forever to input all those parlays into betmmaztips and the bookie but maybe once | automate the placing
of pre-event picks into betmma.tips or something then I'll start doing all parlay combinations possible as
the recommended picks.

71/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

New Key Feature: Adjusted Performance

January 11, 2025

Alright, so I'm finally ready to talk about my new key feature: Adjusted Performance. This has been one «
those "why didn't | think of this sooner" moments, but also "holy crap this is a nightmare to implement."
Let me give you a quick taste of what it looks like mathematically:

fighter1_stat_adjperf = (fighter1_stat - fighter2_stat_opp_avg) / fighter2_stat_opp_sdev

We'll call this f1_stat_adjperf for short. The big idea is to measure how much a fighter's performance in
any given fight exceeded or fell short of what we'd expect their opponent to allow. If you go in there
against some unstoppable jab machine who normally forces everyone to eat 50 jabs per round, but you
manage to hold them to only 20, well, that's huge. But if you never do the math to figure out what your
opponent's "baseline" is, you'll just record "Fighter1 absorbed 20 jabs" and maybe think it's not that
great. Meanwhile, that's incredibly good compared to the 50 jabs everyone else took. Hence, adjusted
performance.

Understanding the _opp Suffix

Before going any further, let's talk about the _opp suffix. First, _opp is the post-fight stats your opponer
did against you. So if you see something like f2_stat_opp_avg, that means the stat is referencing "what
your opponent's opponents did against them." For example:

* f1_stat_opp_avg: The average of your opponents’ performances against you in all their fights.
+ {2_stat_opp_avg: The average of your opponent's opponents’ performances against your opponent
in all their previous fights.

* £2_stat_opp_sdev: The standard deviation of your opponent's opponents performance in those sar
fights, giving us a measure of the volatility or variability in their performance.

So, if your opponent has 40 strikes landed against them then f2_strikes_opp_avg is 40.

This means, to measure your adjusted performance, you compare your post-fight stat in the fight to the
pre-fight opp_avg, and then scale it by pre-fight _opp_sdev. If you get 60 strikes against someone whc
averages getting hit 30 times, you're 1 standard deviation above average against that person.

Why It's So Valuable

Let's be honest: raw stats can lie, or at least mislead. If a fighter's output is just "I landed 30 strikes," the
doesn't tell me how many strikes they should have been able to land. If they were fighting an iron-clad
defensive wizard who typically only allows 10 strikes, then landing 30 is insane. On the flip side, if your
opponent is basically an open punching bag who gives up 60 strikes on average, then landing 30 is
actually pretty weak.

Adjusted performance changes the game: it says, "30 strikes might be good or bad, but let's see how it
compares to what your opponent usually allows." Then, for even more nuance, it's scaled by the
opponent's historical standard deviation—so you don't artificially inflate your stats just because you face
someone with wide variability on a certain stat.

The Complexity: Where Do You Even Get These Numbers?

The problem with pulling off something like f|_stat_adjperf is that you actually have to calculate your
opponent's _opp_avg and _opp_sdev. In other words:

72179


1/6/26, 9:08 AM MMA-ALnet
* Grab all your opponent's previous fights.
+ For each of those fights, figure out the stats they allowed to their opponents.
+ From that, compute the average allowed stats and the standard deviation of those stats.

+ Then bring that back into your current fight to see how well or poorly you did.

This is where the data pipeline can get insane, because you might have a fighter who has 15 or 20 fights
each with a different set of opponents. Some of those opponents have 30 fights apiece. Doing this for
every fighter means you need to traverse a huge web of fight stats.

If you run a naive approach—tike just using Panda's groupby and merges all day—it becomes unbelievab
slow at scale. This is why | had to do some major refactoring, rewriting a chunk of my data pipeline to pu
from a properly indexed database (Postgres, in my case). Once your data is properly structured, it's muc
faster to do these calculations in a single pass or via specialized queries, rather than stumbling around it
memory merges.

Time-Decayed Averages & Time-Decayed StdDev

But wait—there's more. | decided to do a time-decayed average (and corresponding standard deviation)
witha 1.5-year half-life. That means that a fight 3 months ago is given a lot more weight than a fight 5
years ago, which is basically ancient history in fight years.

Now the complexity is multiplied by, like, a factor of 10. Because to get f2_stat_opp_avg, I can't just
average everything your opponent has done; | have to:

Grab each fight's stat.

Weight it exponentially based on how recently it happened.

Sum up those weighted stats.

Divide by the sum of the weights.

Then do it all over again for the standard deviation.

And let's not forget: we do this for every single fight, across thousands of fights, across hundreds of
fighters. That's why | always say data engineering is half the battle.

Why Bother?

So why go through this code-wrangling fiasco when your standard raw stats might be "good enough?"
Because fights are context-dependent. If you have a stand-up specialist with insane takedown defense,
but no one has tested it in years, your raw stats might not reflect the real picture of how they handle a
brand-new style. By blending in adjusted performance stats, you're no longer stuck just describing how
many strikes or takedowns a fighter landed; you're describing how well they did compared to what their
opponent typically experiences—and you're discounting or boosting older fights according to their
recency.

This is how we start to see nuanced differences that no raw stat alone can show. It's the difference
between "Fighter A landed 40 strikes" vs. "Fighter A forced a 1.5 standard-deviation drop in Fighter B's
typical striking output." That second statement captures so much more power. If you can integrate thes«
insights into your model, you get a far more realistic prediction of how a matchup might turn out.

https://www.mma-ai.net/news 73/79


1/6/26, 9:08 AM MMA-ALnet

MMA-AI.net v5 New Years Updates

December 27, 2024

V5 beta is *might* be done in time for the next event. I'm 100's of hours in. The data processing and
model training is basically done, | just need to figure out the final feature set and do some tuning. Then |
need to write the future prediction code for creating and cleaning the data of future fights. Finally, we're
beating Vegas accuracy/log loss without including the odds or rating/ranking stats like an Elo score. Tha
not to say | won't include those in the future, but it's a great sign that the fundamentals of the model are
improving. | would like to thank my $200/mo ChatGPT o1 Pro subscription for making this possible. That
model absolutely rules at complex code, especially statistics and math.

Here's a recent training run to give you an idea of current levels of performance. This isn't the final mod
performance, it's just from tinkering with the feature set and training parameters:

Model Performance:
Training log loss: -0.611
Validation log loss: -0.575
Test Log loss: -0.612
Training accuracy: 0.671
Validation accuracy: 0.698

Test accuracy: 0.691

Basically, we're seeing somewhere around 68% to 69% accuracy on last year's fights (Vegas is current!
about 67% accurate) that the model has never seen before with excellent log loss. Log loss is the ability
the model to accurately predict the chance a fighter will win. For example, if in its training data it predict
50 fighters to have a 70% chance to win, and those fighters won 68% of the time, then the log loss is
nicely calibrated. This is what allows us to check the EV of Al predictions versus Vegas odds.

Changes:
* Total rewrite

© Switched from just pandas dataframes (SLOW) to Postgres SQL database (FAST)

© Greatly improved code readability, design, and efficiency for easier future updates
+ Features

© Total feature overhaul

© Per minute
= Uses Bayesian Posterior Mean to smooth the outliers, zeros, and noise reduction
© Accuracy/defense

= Uses Bayesian smoothing with a Beta prior to smooth the outliers, zeros, and noise reducti

« Priors are calculated on a historical pre-fight per-weightclass, per-stat basis
© Ratio

= Now does bounded ratio to prevent division by zero and outlier ratios

https://www.mma-ai.net/news 74/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

© Average

= Now includes time decayed average rather than recent average

The reason we do all this smoothing is because when a fighter attempts 0 submissions and lands 0
submissions, giving them 0% accuracy is an unrealistic measure of what their submission accuracy
would've been had they attempted any submissions. With smoothing it looks more like this:

Subs_land / subs_attempted = accuracy
0/0 = 20% acc (similar to historical weightclass average submission accuracy)
0/1=18% ace

0/10=
These are just examples, not the actual numbers but you get the idea. It punishes fighters who never lar
any attempts but doesn't overpunish them which means future stats that are layered on top will be more
realistic to their actual performance based on sparse data.

% acc

Second, no more recent averages over an arbitrary number of fights or dates. Time decayed averages a
where it's at. Set a half life, like 1 year. Fights within this 1 year account for, let's say, 50% of the time
decayed average. Fights the year prior account for 25% of the time decayed average, etc. This gives a
much more precise measure of the fighter as he stands today rather than letting how he stood 5 years
ago affect his current stats.

All of these changes above were nice and showed a moderate increase in the model's reliability. Howeve
the real coup de grace was a final layer of statistical analysis over the stats that solved the following
problem: if you fought nothing but cans 20 fights in a row your stats would make you look better than Jo
Jones. But if you fought nothing but top flight competition 20 fights in a row, your stats would look
average to below average despite the fact you'd crush the can crusher. The main way people have solve
this has been ranking or rating systems like Elo scores. While this is pretty effective it doesn't solve the
core problem of the fighter's individual fight stats being a kind of isolated measurement of the fighter th
lacks perspective. How do you turn these core, fight by fight stats into an interconnected web where ea
individual fight's stats can inform and affect the stats of other related fights?

I'll go into detail about the exact math | did to solve that problem along with the mathematical

implementation of the Bayesian smoothing on my Patreon for subscribers here soon. See you next event
hopefully!

75/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

The Art of Not Sucking at Al: A Post-Mortem
of My Model's Spectacular Face-Plant

March 24, 2024

After watching UFC 309 systematically demolish my model's predictions with the precision of a prime
Anderson Silva, I've spent the past week in a caffeine-fueled debugging frenzy. Between muttering
obscenities at my terminal and questioning my life choices, I've had some genuine epiphanies about Al
development that might save others from my particular flavor of statistical hell.

The Problem with Yes-Man Al

Large Language Models are like that friend who encourages your 3 AM business ideas. "A blockchain-
based dating app for cats? Brilliant!" This becomes particularly dangerous when you're knee-deep in
feature engineering and looking for validation rather than criticism.

After extensive testing, I've discovered something fascinating about GPT o1's mathematical capabilities.
While most LLMs give basic statistical approaches, GPT 01 can dive deep into complex statistical
problems. But the real breakthrough came from building an Al feedback loop: get statistical approaches
from GPT 01, feed them to Claude for implementation (it writes cleaner code), then feed Claude's code
back to GPT o1 for validation.

Even debugging has improved. When Claude's code throws an exception, feeding the error back works
once. But for persistent issues, asking Claude "what do you need to debug this error?" is far more
effective. It responds with diagnostic code that, once fed with real data, leads to actual fixes rather than
band-aids.

This iterative process, combined with extensive prompt engineering and lots of sample data to help GPT
01 truly understand the problem domain, has led to the first major mathematical breakthrough in V5's
development: our new Bayesian approach to handling fight statistics.

Bayesian Beta Binomial: The Zero Division Solution

This is only one of many many improvements in V5 but | find it super interesting so I'm writing about it at
fuck you if you don't want to hear about it. Let's dive deep into how we handle the dreaded divide-by-ze
problem in fight statistics. When calculating success rates (like submission accuracy or strike accuracy)
we use Bayesian Beta Binomial analysis to provide meaningful priors that smoothly handle edge cases.

The approach works like this: Instead of naive division that breaks on zeros, we model each ratio as a Be
distribution where:

* a (alpha) represents successful attempts plus prior successes
* B (beta) represents failed attempts plus prior failures

+ The posterior mean (a / (a + B)) gives us our smoothed estimate
For example, with submission accuracy:
submission_accuracy = (submissions + a) / (submission_attempts + a + B)

We determine our priors (a, 8) through empirical Bayes by analyzing the historical distribution of succes
rates across all fighters. These priors vary by stat type, reflecting the different base rates we see in MM,

76/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

+ Submissions: Lower a and higher values reflecting their relative rarity
* Strikes: More balanced a and 8 values reflecting higher occurrence rates

+ Takedowns: Intermediate values based on historical success rates
This approach elegantly handles three critical cases:

* Zero attempts: Returns the prior mean (a/(a+B))
+ Small sample sizes: Heavily weights the prior

* Large sample sizes: Converges to the empirical rate

To understand why this matters, consider how submission accuracy was traditionally handled: A fighter
attempting 10 submissions and landing none would be assigned 0% accuracy. This creates two problem
it skews averages downward, and when comparing fighters (fighter1_sub_acc / fighter2_sub_acc), we ri
another divide-by-zero error.

Our Bayesian approach instead provides more nuanced, realistic estimates. For example:

* 10 attempts, 0 successes = 3.5% accuracy
* Qattempts, 0 successes = 3.8% accuracy

* 8 attempts, 0 successes = 4.1% accuracy

This prevents over-punishing unsuccessful attempts while ensuring we never hit true zero. The accurac\
gradually increases as sample size decreases, reflecting our increasing uncertainty with smaller sample
sizes.

The V5 Redemption Arc

For V5, we're continuing to embrace AutoML (specifically AutoGluon) to eliminate the uncertainty in moc
optimization. Through V1-V3, | spent countless hours manually tuning gradient boosted algorithms like
XGBoost and CatBoost. While | learned an enormous amount about hyperparameter optimization and
various other tuning, I was never entirely confident | was meeting professional machine learning
engineering standards.

AutoML removes that uncertainty. It systematically explores model architectures and hyperparameters it
ways that would take me months to do manually. | still do significant tuning, but nowit's faster and more
reliable. No more wondering if | missed some crucial optimization or made rookie mistakes in the model
architecture.

What This Means For Users

MMA-ALnet will continue hosting Model V4 predictions while V5 is under development. However, there
won't be any improvements to the current model during this transition. If you want to take a step back at
not ride with me for a month or two while | finish V5, | completely understand. This isn't about quick fixe:
it's about building something that actually works.

The Bottom Line

Sometimes you need to lose six parlays in a row to light a fire under your ass and actually learn some
math. But at least we're failing forward, and V5 will be built on more solid foundations.

P.S. To the UFC 309 fighters: those 3 AM tweets weren't personal. It was just the model and the Monste
Energy talking.

7179


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Success! Then Failure?

March 20, 2024

I'm absolutely furious. UFC 309 just wiped out 6 units with 6 consecutive parlay losses. After four years
development, countless sleepless nights, and what | thought was a breakthrough in betting strategy,
reality hit hard.

The Journey to Version 4

The evolution of MMA-Al.net has been a marathon. Version 1 took two years of meticulous development
When ChatGPT emerged, | saw an opportunity to rebuild everything from the ground up. Version 3
emerged after three months of intense work—t'm talking 5 AM bedtimes, night after night.

Then came Version 4, inspired by Chris from wolftickets.ai. His analysis revealed something fascinating:
parlays could be more profitable than single picks when accounting for odds. This insight led to another
complete overhaul, this time leveraging autoML to eliminate inefficiencies and using ChatGPT for
guidance. The results were stunning: 50-60% ROI over six months.

Success Breeds Complacency

Those six months of success? They made me soft. | coasted. My initial $270 investment had grown to
$13,000 purely through profit reinvestment—no additional capital needed. The model was working so we
that | focused instead on rebuilding this website using Cursor, an Al-powered IDE. Despite knowing
nothing about web development, HTML, or CSS, | managed to transform MMA-Al.net from an ad-riddlec
eyesore into what you see today.

Traffic surged to all-time highs. Everything seemed perfect. Then UFC 309 happened.

The Wake-Up Call

Six parlays. Six losses. 100% drawdown. The rage I'm feeling isn't just about the money—it's about the
realization that I've been resting on my laurels. Those six months of coasting caught up with me in the
most painful way possible.

This isn't the end of MMA-Alnet. Honestly, it's just what | needed to get my ass in gear and learn some
math. I'm already planning the next evolution.

Stay tuned for my next post about where we go from here.

78/79


1/6/26, 9:08 AM

https://www.mma-ai.net/news

MMA-ALnet

Welcome to the new MMA-AI.net

March 19, 2024

After three years of development, thousands of hours of feature engineering, and endless testing, | final
redid the stupid website. Goodbye ads, hello pretty new design. This platform represents what | believe
be one of the most sophisticated sports prediction models available today.

The past 5 months have been particularly exciting as we've cracked one of the final pieces of the profit-
maximization puzzle: the betting strategy. Chris from wolftickets.ai and | knew there had to be a better
approach than simply betting straight picks to maximize ROI. Chris was the first to discover it: parlays.

Since both of our models maintain significant accuracy and log loss advantages over Vegas, we can
multiply that edge using parlays. Why? Because parlay odds aren't additive—they're multiplicative.
Through extensive testing, we've found that 3-leg parlays offer the optimal balance of risk and reward.
While 4-leg parlays actually showed higher ROI in testing, their boom-or-bust nature led to more extrerr
bankroll swings and higher bankruptcy risk. Since implementing the 3-leg strategy five months ago, the
model has achieved a 50% ROI.

The Parlay Strategy

Our approach is straightforward: randomly selected Al picks combined into parlays, with no fighter
appearing more than twice to prevent single-fighter dependency. The most common question | get is
"Why not just parlay the +EV fighters together?" Well, I've tested hundreds of parlay permutations:
underdogs only, favorites only, +EV only, 1-5 leg combinations—every variation imaginable—against a ye
of unseen fight data. Surprisingly, the +EV-only strategies consistently underperformed compared to
randomly selected Al pick parlays.

This might seem counterintuitive, but it likely stems from how we're solving a binary classification
problem. Our models excel at distinguishing wins (1) from losses (0), but may be less refined at setting
precise win probabilities. | train on log loss, not accuracy—log loss being a metric that heavily penalizes
confident mistakes while rewarding confident correct predictions. You can see evidence of this in the
calibration curve on our About page.

But honestly? | don't care about the "why." Too many people get tunnel vision focusing on EV, which
makes sense if you're working without a mathematical model like MMA-Al.net. But when you have a
proven model with demonstrable advantages over Vegas odds, the only metric that matters is ROI. And
our testing shows that random Al pick parlays consistently deliver the highest returns.

So here's to the new site, the new strategy, and the new model. You can find my predictions posted her«
on the home page and occasionally on Twitter/X or https://reddit.com/r/mmabetting before each event.

79/79
