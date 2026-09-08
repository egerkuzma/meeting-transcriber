import ApplicationServices
@testable import MeetingTranscriber
import XCTest

/// Accessibility permission verdicts, split out of `PermissionHealthCheckTests`
/// so neither class outgrows the 400-line body limit. The probe's three-way
/// outcome carries most of the reasoning in this area, so it earns its own file.
final class PermissionHealthCheckAccessibilityTests: XCTestCase {
    func testAccessibilityHealthy() {
        let result = PermissionHealthCheck.checkAccessibility(trusted: true, probe: .responded)
        XCTAssertEqual(result, .healthy)
    }

    func testAccessibilityDenied() {
        let result = PermissionHealthCheck.checkAccessibility(trusted: false, probe: .apiDisabled)
        XCTAssertEqual(result, .denied)
    }

    func testAccessibilityDeniedEvenIfProbeSucceeds() {
        // Defensive: if the system says no, we never report healthy.
        let result = PermissionHealthCheck.checkAccessibility(trusted: false, probe: .responded)
        XCTAssertEqual(result, .denied)
    }

    func testAccessibilityBroken() {
        let result = PermissionHealthCheck.checkAccessibility(trusted: true, probe: .apiDisabled)
        XCTAssertEqual(result, .broken)
    }

    /// The regression this split exists for. `.cannotComplete` is what a
    /// frontmost app that does not answer AX produces, and it used to collapse
    /// into the same "broken" as a genuinely closed API — which is how a healthy
    /// grant got a notification telling the user to toggle it off and on.
    func testAccessibilityInconclusiveProbeIsNotReportedAsBroken() {
        let result = PermissionHealthCheck.checkAccessibility(
            trusted: true, probe: .inconclusive(.cannotComplete),
        )
        XCTAssertEqual(
            result, .healthy,
            "An unresponsive frontmost app says nothing about our own grant",
        )
    }

    /// Every non-`apiDisabled` error takes the inconclusive path, so the fix
    /// does not depend on guessing which code a busy app happens to return.
    func testAccessibilityOtherAXErrorsAreAlsoInconclusive() {
        for error in [AXError.failure, .invalidUIElement, .noValue, .attributeUnsupported] {
            XCTAssertEqual(
                PermissionHealthCheck.checkAccessibility(trusted: true, probe: .inconclusive(error)),
                .healthy,
                "AXError \(error.rawValue) describes the element asked, not our permission",
            )
        }
    }

    /// `.notImplemented` sits next to `.apiDisabled` in the AX header and means
    /// something entirely different: the *asked* process has incomplete AX
    /// support. Classifying it as a permission failure would re-create the false
    /// alarm for every such app, so the probe must keep them apart.
    func testAccessibilityProbeTreatsNotImplementedAsInconclusive() {
        XCTAssertEqual(
            PermissionHealthCheck.checkAccessibility(trusted: true, probe: .inconclusive(.notImplemented)),
            .healthy,
        )
        XCTAssertEqual(
            PermissionHealthCheck.checkAccessibility(trusted: true, probe: .apiDisabled),
            .broken,
            "The one error that IS about our access still reports broken",
        )
    }

    /// The verdict alone could not explain the false alarm; the raw code can.
    func testAccessibilityProbeLogTokenCarriesTheRawErrorCode() {
        XCTAssertEqual(
            PermissionHealthCheck.AccessibilityProbe.inconclusive(.cannotComplete).logToken,
            "inconclusive(AXError \(AXError.cannotComplete.rawValue))",
        )
        XCTAssertEqual(PermissionHealthCheck.AccessibilityProbe.responded.logToken, "responded")
    }
}
