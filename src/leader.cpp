/*
 * leader.cpp (wrist-enabled)
 */
#include "external_torque.h"
#include <iostream>
#include <string>
#include <boost/thread.hpp>
#include <barrett/detail/stl_utils.h>
#include <barrett/os.h>
#include <barrett/products/product_manager.h>
#include <barrett/systems.h>
#include <barrett/units.h>

#define BARRETT_SMF_VALIDATE_ARGS
#include <barrett/standard_main_function.h>
#include "ros/ros.h"

#include <haptic_wrist/haptic_wrist.h>

#include "leader.h"                     // wrist-enabled header above
#include "background_state_publisher.h"
#include "leader_dynamics.h"
#include "dynamic_external_torque.h"

using namespace barrett;
using detail::waitForEnter;

void printUsage(const std::string& programName, const std::string& remoteHost, int recPort, int sendPort) {
    std::cout << "Usage: " << programName << " [remoteHost] [recPort] [sendPort]\n";
    std::cout << "       Defaults: remoteHost=" << remoteHost << ", recPort=" << recPort << ", sendPort=" << sendPort << "\n";
    std::cout << "       -h or --help: Display this help message.\n";
}

bool validate_args(int argc, char** argv) {
    if ((argc == 2 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) || (argc > 4)) {
        printUsage(argv[0], "127.0.0.1", 5555, 5554);
        return 0;
    }
    return true;
}

template <size_t DOF>
int wam_main(int argc, char **argv, ProductManager &pm, systems::Wam<DOF> &wam) {
    BARRETT_UNITS_TEMPLATE_TYPEDEFS(DOF);

    jp_type SYNC_POS;
    if (DOF == 4) {
        SYNC_POS[0] = 0.0;
        SYNC_POS[1] = -1.95;
        SYNC_POS[2] = 0.0;
        SYNC_POS[3] = 2.97;
    } else {
        printf("Error: 4 DOF supported\n");
        return false;
    }

    std::string remoteHost = "127.0.0.1";
    int rec_port = 5555;  // leader receives follower's 5555
    int send_port = 5554; // leader sends on 5554

    if (argc >= 2) remoteHost = std::string(argv[1]);
    if (argc >= 3) rec_port   = std::atoi(argv[2]);
    if (argc >= 4) send_port  = std::atoi(argv[3]);

    ros::init(argc, argv, "leader");
    haptic_wrist::HapticWrist hw;
    hw.gravityCompensate(true);
    hw.run();

    BackgroundStatePublisher<DOF> state_publisher(pm.getExecutionManager(), wam, &hw);

    barrett::systems::Summer<jt_type, 3> customjtSum;
    pm.getExecutionManager()->startManaging(customjtSum);

    LeaderDynamics<DOF> leaderDynamics(pm.getExecutionManager());
    ExternalTorque<DOF> externalTorque(pm.getExecutionManager());
    DynamicExternalTorque<DOF> dynamicExternalTorque(pm.getExecutionManager());

    barrett::systems::FirstOrderFilter<jt_type> extFilter;
    jt_type omega_p(180.0);
    extFilter.setLowPass(omega_p);
    pm.getExecutionManager()->startManaging(extFilter);

    barrett::systems::FirstOrderFilter<jt_type> dynamicExtFilter;
    dynamicExtFilter.setLowPass(omega_p);
    pm.getExecutionManager()->startManaging(dynamicExtFilter);

    // wire dynamics (same as before)
    ja_type ja; ja.setConstant(0.0);
    systems::Constant<ja_type> zeroAcceleration(ja);
    pm.getExecutionManager()->startManaging(zeroAcceleration);

    // Leader system (now wrist-enabled)
    Leader<DOF> leader(pm.getExecutionManager(), &hw, remoteHost, rec_port, send_port);

    jt_type maxRate; maxRate << 50, 50, 50, 50;
    systems::RateLimiter<jt_type> wamJPOutputRamp(maxRate, "ffRamp");

    // optional prints
    systems::PrintToStream<jt_type> printdynamicextTorque(pm.getExecutionManager(), "dynamicextTorque: ");
    systems::PrintToStream<jt_type> printextTorque(pm.getExecutionManager(), "extTorque: ");
    systems::PrintToStream<jt_type> printdynamicoutput(pm.getExecutionManager(), "dynamicoutput: ");
    systems::PrintToStream<jt_type> printSC(pm.getExecutionManager(), "SC: ");

    // filters for ja (as in your original)
    double h_omega_p = 25.0;
    barrett::systems::FirstOrderFilter<jv_type> hp1;
    hp1.setHighPass(jv_type(h_omega_p), jv_type(h_omega_p));
    systems::Gain<jv_type, double, ja_type> jaWAM(1.0);
    pm.getExecutionManager()->startManaging(hp1);

    barrett::systems::FirstOrderFilter<ja_type> jaFilter;
    ja_type l_omega_p = ja_type::Constant(50.0);
    jaFilter.setLowPass(l_omega_p);
    pm.getExecutionManager()->startManaging(jaFilter);

    systems::connect(wam.jvOutput, hp1.input);
    systems::connect(hp1.output, jaWAM.input);
    systems::connect(jaWAM.output, jaFilter.input);
    // systems::connect(jaFilter.output, leaderDynamics.jaInputDynamics); // keep if you used it before

    systems::connect(wam.jpOutput, leader.wamJPIn);
    systems::connect(wam.jvOutput, leader.wamJVIn);
    systems::connect(wam.gravity.output, leader.wamGravIn);

    systems::connect(wam.jpOutput, leaderDynamics.jpInputDynamics);
    systems::connect(wam.jvOutput, leaderDynamics.jvInputDynamics);

    systems::connect(leaderDynamics.dynamicsFeedFWD, leader.wamDynIn);

    // ext torque path (choose source)
    systems::connect(wam.gravity.output, externalTorque.wamGravityIn);
    systems::connect(wam.jtSum.output,  externalTorque.wamTorqueSumIn);
    systems::connect(externalTorque.wamExternalTorqueOut, extFilter.input);

    // your dynamic ext torque path
    systems::connect(customjtSum.output,              dynamicExternalTorque.wamTorqueSumIn);
    systems::connect(leaderDynamics.dynamicsFeedFWD,  dynamicExternalTorque.wamDynamicsIn);
    systems::connect(dynamicExternalTorque.wamExternalTorqueOut, dynamicExtFilter.input);
    systems::connect(dynamicExternalTorque.wamExternalTorqueOut, leader.extTorqueIn);
    // or: systems::connect(extFilter.output, leader.extTorqueIn);

    // sum torque cmd + gravity + supervisory
    systems::connect(leader.wamJPOutput,                 customjtSum.getInput(0));
    systems::connect(wam.gravity.output,                 customjtSum.getInput(1));
    systems::connect(wam.supervisoryController.output,   customjtSum.getInput(2));
    connect(customjtSum.output, wam.input);

    wam.gravityCompensate();

    std::string line; v_type gainTmp;
    bool going = true;

    while (going) {
        printf(">>> ");
        std::getline(std::cin, line);
        if (line.empty()) { printf("\n"); continue; }

        switch (line[0]) {
        case 'l':
            if (leader.isLinked()) {
                leader.unlink();
            } else {
                wam.moveTo(SYNC_POS, true);
                hw.setTarget({0.0, 0.0, 0.0});  // no brace init

                printf("Press [Enter] to link with the other WAM.");
                waitForEnter();
                leader.tryLink();
                wam.trackReferenceSignal(leader.theirJPOutput);
                connect(leader.wamJPOutput, wam.input);

                // connect(leader.wamJPOutput, wamJPOutputRamp.input);
                // connect(wamJPOutputRamp.output, wam.input);

                btsleep(0.1);
                if (leader.isLinked()) printf("Linked with remote WAM.\n");
                else                    printf("WARNING: Linking was unsuccessful.\n");
            }
            break;

        case 't': {
            size_t jointNumber;
            std::cout << "\tJoint: ";
            std::cin >> jointNumber;
            size_t jointIndex = jointNumber - 1;
            if (jointIndex >= DOF) { std::cout << "\tBad joint number: " << jointNumber; break; }

            char gainId;
            std::cout << "\tGain identifier (p, i, or d): ";
            std::cin >> line; gainId = line[0];

            std::cout << "\tCurrent value: ";
            switch (gainId) {
            case 'p': gainTmp = wam.jpController.getKp(); break;
            case 'i': gainTmp = wam.jpController.getKi(); break;
            case 'd': gainTmp = wam.jpController.getKd(); break;
            default:  std::cout << "\tBad gain identifier.";
            }
            std::cout << gainTmp[jointIndex] << std::endl;

            std::cout << "\tNew value: ";
            std::cin >> gainTmp[jointIndex];
            switch (gainId) {
            case 'p': wam.jpController.setKp(gainTmp); break;
            case 'i': wam.jpController.setKi(gainTmp); break;
            case 'd': wam.jpController.setKd(gainTmp); break;
            default:  std::cout << "\tBad gain identifier.";
            }
        } break;

        case 'x':
            going = false; break;

        default:
            printf("\n");
            printf("    'l' to toggle linking with other WAM (and wrist)\n");
            printf("    't' to tune control gains\n");
            printf("    'x' to exit\n");
            break;
        }
    }

    pm.getSafetyModule()->waitForMode(SafetyModule::IDLE);
    hw.stop();
    return 0;
}

