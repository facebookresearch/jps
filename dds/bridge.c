/*
 * DOUBLE_DUMMY
 * Copyright (c) 1990, 2000 by James D. Allen (jamesdowallen@yahoo.com)
 *
 * This program will solve double-dummy bridge problems.
 * The algorithm is trivial: brute-force alpha-beta search (also known
 *      as "backtracking").  The alpha-beta is especially easy here since
 *      we do not consider overtricks or extra undertricks.
 * The control flow is ``way cool'': this is a rare exception where software
 *      is more readable with a "goto".  (Although I've tersified this to
 *      the point where it is perhaps unreadable anyway :-)
 * Yes, the `goto' won't operate properly unless all the local variables
 *      are in precisely the right state to make their strange voyage from
 *      one inner loop to the other.  But that's exactly the concept which
 *      the potential code-maintainer is intended to grasp.
 *
 * This "dumb" program could be sped up considerably but, as it is, it
 *      does solve *one* complete 52-card position ("13-card ending")
 *      in about 1 minute that many expert humans cannot solve at all.
 *      That special position is sort of a quirk.
 *
 * This program is "copylefted."  You are authorized to use or reproduce it
 *      as long as you observe all three of the following admonitions:
 *   (1)  Do not sell this software for profit.
 *   (2)  Do not remove the Copyright notice or these admonitions.
 *   (3)  Do not use an alternate construction which avoids the `goto'  :-}
 */

#define NUMP    4       /* The Players:  N, E, S, W */
#define         NORTH   0
#define         IS_CONT(x)      (!((x) & 1))    /* Is x on N/S team? */
#define         LHO(x)          (((x) + 1) % NUMP)
#define         RHO(x)          (((x) + NUMP - 1) % NUMP)
char    *Playername[] = { "North", "East", "South", "West" };

#define NUMS    4       /* The Suits:   S, H, D, C */
char    Suitname[] = "SHDC";
char    *Fullname[] = { "Spades  ", "Hearts  ", "Diamonds", "Clubs   ", };

/*
 * Rank indices are 2 (Ace), 3 (King), ... 14 (Deuce)
 * There is also a CARD Index which can be converted to from Rank and Suit.
 */
#define CARD(Suit, Rank)        (((Suit) << 4) | (Rank))
#define SUIT(Card)              ((Card) >> 4)
#define RANK(Card)              ((Card) & 0xf)
char    Rankname[] = "??AKQJT98765432";
#define INDEX(s, c)     ((char *)strchr(s, c) - (s))

/* A "SuitSet" is used to cope with more than one card at once: */
typedef unsigned short SuitSet;
#define MASK(Card)              (1 << RANK(Card))
#define REMOVE(Set, Card)       ((Set) &= ~(MASK(Card)))

/* And a CardSet copes with one SuitSet for each suit: */
typedef struct cards {
        SuitSet cc[NUMS];
} CardSet;

/* Everything we wish to remember about a trick: */
struct status {
        CardSet Holding[NUMP];  /* The players' holdings */
        CardSet Legal[NUMP];    /* The players' remaining legal plays */
        short   Played[NUMP];   /* What the players played */
        SuitSet Trick;          /* Led-suit Cards eligible to win trick */
        SuitSet Trump;          /* Trump Cards eligible to win trick */
        short   Leader;         /* Who led to the trick */
        short   Suitled;        /* Which suit was led */
} Trickinfo[14]; /* Status of 13 tricks and a red zone" */

highcard(set)
        SuitSet set;
{
        return set & 0xff ? set &  1 ? 0 : set &  2 ? 1 : set &  4 ? 2
                          : set &  8 ? 3 : set & 16 ? 4 : set & 32 ? 5
                          : set & 64 ? 6 : 7 : highcard(set >> 8) + 8;
}

main()
{
        register struct status *P = Trickinfo;  /* Point to current status */
        int     tnum;   /* trick number */
        int     won;    /* Total tricks won by North/South */
        int     nc;     /* cards on trick */
        int     ohsize; /* original size of hands */
        int     mask;
        int     trump;
        int     player; /* player */
        int     pwin;   /* player who won trick */
        int     suit;   /* suit to play */
        int     wincard; /* card which won the trick */
        int     need;   /* Total tricks needed by North/South */
        int     cardx;  /* Index of a card under consideration */
        int     success; /* Was the trick or contract won by North/South ? */
        int     last_t; /* Decisive trick number */
        char    asciicard[60];
        SuitSet inplay; /* cards still in play for suit */
        SuitSet pr_set; /* Temp for printing */

        /* Read in the problem */
        printf("Enter trump suit (0-S,1-H,2-D,3-C,4-NT): ");
        scanf("%d", &trump);
        printf("Enter how many tricks remain to be played: ");
        scanf("%d", &ohsize);
        printf("Enter how many tricks North/South need to win: ");
        scanf("%d", &need);
        printf("Enter who is on lead now (0=North,etc.): ");
        scanf("%d", &pwin);
        printf("Enter the %d cards beginning with North:\n", NUMP * ohsize);
        for (player = NORTH; player < NUMP; player++) {
                for (tnum = 0; tnum < ohsize; tnum++) {
                        scanf("%s", asciicard);
                        cardx = CARD(INDEX(Suitname, asciicard[1]),
                                        INDEX(Rankname, asciicard[0]));
                        P->Holding[player].cc[SUIT(cardx)] |= MASK(cardx);
                }
        }

        /* Handle the opening lead */
        printf("Enter the directed opening lead or XX if none:\n");
        P->Legal[pwin] = P->Holding[pwin];
        scanf("%s", asciicard);
        if (asciicard[0] == 'X') {
                strcpy(asciicard, "anything");
        } else {
                cardx = CARD(INDEX(Suitname, asciicard[1]),
                                INDEX(Rankname, asciicard[0]));
                for (suit = 0; suit < NUMS; suit++)
                        if (suit != SUIT(cardx))
                                P->Legal[pwin].cc[suit] = 0;
                        else if (!(P->Legal[pwin].cc[suit] &= MASK(cardx))) {
                                printf("He can't lead card he doesn't have\n");
                                exit(1);
                        }
        }

        /* Print the problem */
        for (player = NORTH; player < NUMP; player++) {
                printf("\n---- %s Hand ----:\n", Playername[player]);
                for (suit = 0; suit < NUMS; suit++) {
                        printf("\t%s\t", Fullname[suit]);
                        for (pr_set = P->Holding[player].cc[suit]; pr_set;
                                        REMOVE(pr_set, highcard(pr_set)))
                                printf("%c ", Rankname[RANK(highcard(pr_set))]);
                        printf("\n");
                }
        }
        printf("\n%s%s Trumps; %s leads %s; N-S want %d tricks; E-W want %d\n",
                trump < NUMS ? Fullname[trump] : "",
                trump < NUMS ? " are" : "No",
                Playername[pwin], asciicard, need, ohsize + 1 - need);

      /* Loop to play tricks forward until the outcome is conclusive */
        for (tnum = won = success = 0;
                        success ? ++won < need : won + ohsize >= need + tnum;
                        tnum++, P++, success = IS_CONT(pwin)) {
                for (nc = 0, player = P->Leader = pwin; nc < NUMP;
                                        nc++, player = LHO(player)) {
                      /* Generate legal plays except opening lead */
                        if (nc || tnum)
                                P->Legal[player] = P->Holding[player];
                      /* Must follow suit unless void */
                        if (nc && P->Legal[player].cc[P->Suitled])
                                for (suit = 0; suit < NUMS; suit++)
                                        if (suit != P->Suitled)
                                                P->Legal[player].cc[suit] = 0;
                        goto choose_suit; /* this goto is easily eliminated */
                      /* Comes back right away after choosing "suit"  */
                choose_card:
                        P->Played[player] = cardx =
                                CARD(suit, highcard(P->Legal[player].cc[suit]));
                        if (nc == 0) {
                                P->Suitled = suit;
                                P->Trick = P->Trump = 0;
                        }
                      /* Set the play into Trick if it is win-eligible */
                        if (suit == P->Suitled)
                                P->Trick |= MASK(cardx);
                        if (suit == trump)
                                P->Trump |= MASK(cardx);

                      /* Remove card played from player's holding */
                        (P+1)->Holding[player] = P->Holding[player];
                        REMOVE((P+1)->Holding[player].cc[suit], cardx);
                }

              /* Finish processing the trick ... who won? */
                if (P->Trump)
                        wincard = CARD(trump, highcard(P->Trump));
                else
                        wincard = CARD(P->Suitled, highcard(P->Trick));
                for (pwin = 0; P->Played[pwin] != wincard; pwin++)
                        ;
        }

      /* Loop to back up and let the players try alternatives */
        for (last_t = tnum--, P--; tnum >= 0; tnum--, P--) {
                won -= IS_CONT(pwin);
                pwin = P->Leader;
                for (player = RHO(P->Leader), nc = NUMP-1; nc >= 0;
                                        player = RHO(player), nc--) {
                      /* What was the play? */
                        cardx = P->Played[player];
                        suit = SUIT(cardx);
                      /* Retract the played card */
                        if (suit == P->Suitled)
                                REMOVE(P->Trick, cardx);
                        if (suit == trump)
                                REMOVE(P->Trump, cardx);
                      /* We also want to remove any redundant adjacent plays */
                        inplay =  P->Holding[0].cc[suit] | P->Holding[1].cc[suit]
                                | P->Holding[2].cc[suit] | P->Holding[3].cc[suit];
                        for (mask = MASK(cardx); mask <= 0x8000; mask <<= 1)
                                if (P->Legal[player].cc[suit] & mask)
                                        P->Legal[player].cc[suit] &= ~mask;
                                else if (inplay & mask)
                                        break;
                      /* If the card was played by a loser, try again */
                        if (success ? !(IS_CONT(player)) : IS_CONT(player)) {
                        choose_suit:
                              /* Pick a suit if any untried plays remain */
                                for (suit = 0; suit < NUMS; suit++)
                                        if (P->Legal[player].cc[suit])
                                                /* This goto is really nice!! */
                                                goto choose_card;
                        }
                }
        }

        /*
         * We're done.  We know the answer.
         * We can't remember all the variations; fortunately the
         *  succeeders played correctly in the last variation examined,
         *  so we'll just print it.
         */
        printf("Contract %s, for example:\n",
                        success ? "made" : "defeated");
        for (tnum = 0, P = Trickinfo; tnum < last_t; tnum++, P++) {
                printf("Trick %d:", tnum + 1);
                for (player = 0; player < P->Leader; player++)
                        printf("\t");
                for (nc = 0; nc < NUMP; nc++, player = LHO(player))
                        printf("\t%c of %c",
                                Rankname[RANK(P->Played[player])],
                                Suitname[SUIT(P->Played[player])]);
                printf("\n");
        }
        exit(0);
}
