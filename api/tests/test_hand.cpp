#include <iostream>
#include <cassert>
#include "Hand.h"
#include "Card.h"

void testRoyalFlush() {
    std::cout << "Testing royal flush..." << std::endl;
    
    std::vector<Card> royalFlush = {
        Card("AS"), Card("KS"), Card("QS"), Card("JS"), Card("TS")
    };
    auto hand = Hand::evaluate(royalFlush);
    
    assert(hand.getRankingName() == "Royal Flush");
    std::cout << "  ✓ Royal flush detected correctly" << std::endl;
}

void testStraightFlush() {
    std::cout << "Testing straight flush..." << std::endl;
    
    std::vector<Card> straightFlush = {
        Card("9H"), Card("8H"), Card("7H"), Card("6H"), Card("5H")
    };
    auto hand = Hand::evaluate(straightFlush);
    
    assert(hand.getRankingName() == "Straight Flush");
    std::cout << "  ✓ Straight flush detected correctly" << std::endl;
}

void testFourOfAKind() {
    std::cout << "Testing four of a kind..." << std::endl;
    
    std::vector<Card> fourOfAKind = {
        Card("AS"), Card("AH"), Card("AD"), Card("AC"), Card("KS")
    };
    auto hand = Hand::evaluate(fourOfAKind);
    
    assert(hand.getRankingName() == "Four of a Kind");
    std::cout << "  ✓ Four of a kind detected correctly" << std::endl;
}

void testFullHouse() {
    std::cout << "Testing full house..." << std::endl;
    
    std::vector<Card> fullHouse = {
        Card("AS"), Card("AH"), Card("AD"), Card("KS"), Card("KH")
    };
    auto hand = Hand::evaluate(fullHouse);
    
    assert(hand.getRankingName() == "Full House");
    std::cout << "  ✓ Full house detected correctly" << std::endl;
}

void testFlush() {
    std::cout << "Testing flush..." << std::endl;
    
    std::vector<Card> flush = {
        Card("AS"), Card("KS"), Card("QS"), Card("JS"), Card("9S")
    };
    auto hand = Hand::evaluate(flush);
    
    assert(hand.getRankingName() == "Flush");
    std::cout << "  ✓ Flush detected correctly" << std::endl;
}

void testStraight() {
    std::cout << "Testing straight..." << std::endl;
    
    std::vector<Card> straight = {
        Card("9H"), Card("8S"), Card("7D"), Card("6C"), Card("5H")
    };
    auto hand = Hand::evaluate(straight);
    
    assert(hand.getRankingName() == "Straight");
    std::cout << "  ✓ Straight detected correctly" << std::endl;
}

void testThreeOfAKind() {
    std::cout << "Testing three of a kind..." << std::endl;
    
    std::vector<Card> threeOfAKind = {
        Card("AS"), Card("AH"), Card("AD"), Card("KS"), Card("QH")
    };
    auto hand = Hand::evaluate(threeOfAKind);
    
    assert(hand.getRankingName() == "Three of a Kind");
    std::cout << "  ✓ Three of a kind detected correctly" << std::endl;
}

void testTwoPair() {
    std::cout << "Testing two pair..." << std::endl;
    
    std::vector<Card> twoPair = {
        Card("AS"), Card("AH"), Card("KD"), Card("KC"), Card("QH")
    };
    auto hand = Hand::evaluate(twoPair);
    
    assert(hand.getRankingName() == "Two Pair");
    std::cout << "  ✓ Two pair detected correctly" << std::endl;
}

void testPair() {
    std::cout << "Testing pair..." << std::endl;
    
    std::vector<Card> pair = {
        Card("AS"), Card("AH"), Card("KD"), Card("QC"), Card("JH")
    };
    auto hand = Hand::evaluate(pair);
    
    assert(hand.getRankingName() == "One Pair");
    std::cout << "  ✓ Pair detected correctly" << std::endl;
}

void testHighCard() {
    std::cout << "Testing high card..." << std::endl;
    
    std::vector<Card> highCard = {
        Card("AS"), Card("KH"), Card("QD"), Card("JC"), Card("9H")
    };
    auto hand = Hand::evaluate(highCard);
    
    assert(hand.getRankingName() == "High Card");
    std::cout << "  ✓ High card detected correctly" << std::endl;
}

void testHandComparison() {
    std::cout << "Testing hand comparison..." << std::endl;
    
    std::vector<Card> royalFlush = {
        Card("AS"), Card("KS"), Card("QS"), Card("JS"), Card("TS")
    };
    auto hand1 = Hand::evaluate(royalFlush);
    
    std::vector<Card> pair = {
        Card("AS"), Card("AH"), Card("KD"), Card("QC"), Card("JH")
    };
    auto hand2 = Hand::evaluate(pair);
    
    assert(hand1 > hand2);
    assert(!(hand2 > hand1));
    assert(hand1.getRankingName() == "Royal Flush");
    std::cout << "  ✓ Royal flush > Pair: Correct" << std::endl;
    
    std::vector<Card> fourOfAKind = {
        Card("KS"), Card("KH"), Card("KD"), Card("KC"), Card("AS")
    };
    auto hand3 = Hand::evaluate(fourOfAKind);
    
    assert(hand1 > hand3);
    std::cout << "  ✓ Royal flush > Four of a kind: Correct" << std::endl;
}

void testSevenCardEvaluation() {
    std::cout << "Testing 7-card evaluation..." << std::endl;
    
    // 7 cards where best hand is from hole cards + community
    std::vector<Card> sevenCards = {
        Card("AS"), Card("AH"),  // Hole cards - pair of aces
        Card("KS"), Card("QS"), Card("JS"), Card("TS"), Card("2D")  // Community
    };
    auto hand = Hand::evaluate(sevenCards);
    
    // Should find royal flush (AS KS QS JS TS), not just pair of aces
    assert(hand.getRankingName() == "Royal Flush");
    assert(hand.bestFive.size() == 5);
    std::cout << "  ✓ Best 5 from 7 cards found correctly" << std::endl;
}

void testTiebreakerSameRank() {
    std::cout << "Testing tiebreaker for same rank..." << std::endl;
    
    // Pair of aces with King kicker
    std::vector<Card> hand1Cards = {
        Card("AS"), Card("AH"), Card("KD"), Card("QC"), Card("JH")
    };
    auto hand1 = Hand::evaluate(hand1Cards);
    
    // Pair of aces with Queen kicker
    std::vector<Card> hand2Cards = {
        Card("AD"), Card("AC"), Card("QH"), Card("JS"), Card("TD")
    };
    auto hand2 = Hand::evaluate(hand2Cards);
    
    assert(hand1.getRankingName() == "One Pair");
    assert(hand2.getRankingName() == "One Pair");
    assert(hand1 > hand2);  // King kicker beats Queen kicker
    std::cout << "  ✓ Tiebreaker on kicker works correctly" << std::endl;
}

void testWheelStraight() {
    std::cout << "Testing wheel straight (A-2-3-4-5)..." << std::endl;
    
    // A-2-3-4-5 straight (wheel)
    std::vector<Card> wheel = {
        Card("AS"), Card("2H"), Card("3D"), Card("4C"), Card("5H")
    };
    auto hand = Hand::evaluate(wheel);
    
    assert(hand.getRankingName() == "Straight");
    // In a wheel, 5 is the high card (tiebreaker should be 5, not 14)
    assert(hand.tiebreakers[0] == 5);
    std::cout << "  ✓ Wheel straight (A-2-3-4-5) detected correctly" << std::endl;
}

int main() {
    std::cout << "\n🃏 Hand Evaluation Test Suite" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        testRoyalFlush();
        testStraightFlush();
        testFourOfAKind();
        testFullHouse();
        testFlush();
        testStraight();
        testThreeOfAKind();
        testTwoPair();
        testPair();
        testHighCard();
        testHandComparison();
        testSevenCardEvaluation();
        testTiebreakerSameRank();
        testWheelStraight();
        
        std::cout << "\n✅ All Hand evaluation tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with error: " << e.what() << std::endl;
        return 1;
    }
}

