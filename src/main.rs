#![allow(non_snake_case)]
use pharmsol::*;
use std::vec;

fn main() {
    // Create a subject with an observation at time 12, which the system will be solved for
    let subject = Subject::builder("jamaas")
        .observation(0.0, 0.0, 0)
        .repeat(12, 1.0)
        .observation(0.0, 0.0, 1)
        .repeat(12, 1.0)
        .observation(0.0, 0.0, 2)
        .repeat(12, 1.0)
        .build();

    // Initial parameters
    let SA = 20.0; // Size of pool A
    let SB = 25.0; // Size of pool B
    let QA0 = 6.0; // Initial quantity in pool A
    let QB0 = 9.0; // Initial quantity in pool B
    let QT0 = 15.0; // Initial total quantity
    let FOA = 7.0; // External input flux to pool A

    // Kinetic constants for the HMM model
    let VAB = 18.0; // Maximum velocity from pool A to B
    let VBA = 13.0; // Maximum velocity from pool B to A
    let VBO = 8.0; // Maximum velocity from pool B to output
    let KAB = 0.32; // Affinity constant for flux A to B
    let KBA = 0.36; // Affinity constant for flux B to A
    let KBO = 0.31; // Affinity constant for flux B to output

    // Define the parameters in a vector
    let spp = vec![SA, SB, QA0, QB0, QT0, FOA, VAB, VBA, VBO, KAB, KBA, KBO];

    // Define the ODE system
    let ode = equation::ODE::new(
        |x, p, _t, dx, _rateiv, _cov| {
            // fetch_params! is a macro that fetches the parameters from the parameter vector, and assigns them to the variables
            fetch_params!(
                p, SA, SB, _QA0, _QB0, _QT0, FOA, VAB, VBA, VBO, KAB, KBA, KBO
            );

            // We define x[0] as the state for pool A, and x[1] as the state for pool B
            // We also define x[2] as the state for the total quantity
            let QA = x[0];
            let QB = x[1];

            let con_a = QA / SA; // Concentration in pool A
            let con_b = QB / SB; // Concentration in pool B

            let fab = VAB / (1.0 + (KAB / con_a)); // Flux from A to B
            let fba = VBA / (1.0 + (KBA / con_b)); // Flux from B to A
            let fbo = VBO / (1.0 + (KBO / con_b)); // Flux from B to output

            // The ODEs are defined here
            dx[0] = FOA + fba - fab; // Rate of change for pool A
            dx[1] = fab - fba - fbo; // Rate of change for pool B
            dx[2] = FOA - fbo; // Rate of change for total system
        },
        |p| {
            fetch_params!(p, _ka, _ke, tlag, _v);
            lag! {0=>tlag}
        },
        |_p| fa! {},
        |p, _t, _cov, x| {
            fetch_params!(
                p, _SA, _SB, QA0, QB0, QT0, _FOA, _VAB, _VBA, _VBO, _KAB, _KBA, _KBO
            );

            // In this closure we define the initial conditions for the system
            x[0] = QA0;
            x[1] = QB0;
            x[2] = QT0;
        },
        |x, p, _t, _cov, y| {
            fetch_params!(
                p, SA, SB, _QA0, _QB0, _QT0, _FOA, _VAB, _VBA, _VBO, _KAB, _KBA, _KBO
            );
            // This equation specifies the output, e.g. the measured concentrations
            let QA = x[0];
            let QB = x[1];
            let QT = x[2];

            y[0] = QA / SA;
            y[1] = QB / SB;
            y[2] = QT / (SA + SB);
        },
        (3, 3),
    );

    let op = ode.estimate_predictions(&subject, &spp);
    dbg!(op.flat_predictions());
}
