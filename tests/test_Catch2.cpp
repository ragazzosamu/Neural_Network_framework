
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Verifica di prova", "[prova][sezioni]") {
    int a = 4;
    int b = 7;

    SECTION("Addizione") { REQUIRE((a + b) == 11); }

    SECTION("Prodotto") { REQUIRE((a * b) == 28); }
}
