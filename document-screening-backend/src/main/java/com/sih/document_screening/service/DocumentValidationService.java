package com.sih.document_screening.service;

import com.sih.document_screening.dto.ValidationSummary;
import com.sih.document_screening.model.NationalIdRecord;
import com.sih.document_screening.model.PassportRecord;
import com.sih.document_screening.model.VisaRecord;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

@Service
public class DocumentValidationService {

    public ValidationSummary validate(
            String claimedNationality,
            PassportRecord passport,
            VisaRecord visa,
            NationalIdRecord nationalId) {

        List<String> errors = new ArrayList<>();
        LocalDate today = LocalDate.now();

        // 1. Expiry & Date Boundary Checks
        boolean isExpired = false;
        if (passport.getExpiryDate() != null && passport.getExpiryDate().isBefore(today)) {
            errors.add("Passport has expired on " + passport.getExpiryDate());
            isExpired = true;
        }
        if (visa.getValidUntil() != null && visa.getValidUntil().isBefore(today)) {
            errors.add("Visa has expired on " + visa.getValidUntil());
            isExpired = true;
        }

        // Check for impossible birth date (born in future or > 120 years old)
        if (passport.getDateOfBirth() != null) {
            long age = ChronoUnit.YEARS.between(passport.getDateOfBirth(), today);
            if (age < 0 || age > 120) {
                errors.add("Invalid date of birth: calculated age (" + age + ") is outside realistic human bounds");
            }
        }

        // 2. Nationality Consistency
        boolean isNationalityConsistent = true;
        if (passport.getNationality() != null && !passport.getNationality().equalsIgnoreCase(claimedNationality.trim())) {
            errors.add("Claimed nationality (" + claimedNationality + ") mismatches passport nationality (" + passport.getNationality() + ")");
            isNationalityConsistent = false;
        }

        // 3. Cross-Document Name Integrity & Character Substitution Detection
        boolean isNameConsistent = true;
        if (passport.getFullName() != null && nationalId.getFullName() != null) {
            String cleanPassportName = normalize(passport.getFullName());
            String cleanIdName = normalize(nationalId.getFullName());

            if (!cleanPassportName.equals(cleanIdName)) {
                if (isCharacterSubstitutionAttempt(cleanPassportName, cleanIdName)) {
                    errors.add("Suspicious character substitution detected between Passport name and National ID name");
                } else {
                    errors.add("Name mismatch between Passport (" + passport.getFullName() + ") and National ID (" + nationalId.getFullName() + ")");
                }
                isNameConsistent = false;
            }
        }

        // 4. Cross-Document DOB Consistency
        boolean isDobConsistent = true;
        if (passport.getDateOfBirth() != null && nationalId.getDateOfBirth() != null) {
            if (!passport.getDateOfBirth().isEqual(nationalId.getDateOfBirth())) {
                errors.add("Date of birth mismatch between Passport (" + passport.getDateOfBirth() + ") and National ID (" + nationalId.getDateOfBirth() + ")");
                isDobConsistent = false;
            }
        }

        // 5. Visa Type & Stay Duration Boundary Verification
        if (visa.getVisaType() != null) {
            int allowedMaxDays = visa.getVisaType().getDefaultMaxStayDays();
            if (visa.getStayDurationDays() != null && visa.getStayDurationDays() > allowedMaxDays) {
                errors.add("Stay duration (" + visa.getStayDurationDays() + " days) exceeds maximum allowed threshold for "
                        + visa.getVisaType().name() + " (" + allowedMaxDays + " days)");
            }
        } else if (visa.getVisaNumber() != null) {
            errors.add("Unrecognized or missing visa classification type");
        }

        // 6. ICAO 9303 MRZ Integrity
        if (passport.getMrzChecksumValid() != null && !passport.getMrzChecksumValid()) {
            errors.add("ICAO 9303 MRZ Checksum verification failed: Document number, DOB, or Expiry has been altered");
        }

        boolean isValid = errors.isEmpty();

        return new ValidationSummary(
                isValid,
                isNameConsistent,
                isDobConsistent,
                isNationalityConsistent,
                isExpired,errors
        );
    }

    private String normalize(String value) {
        return value.trim().toLowerCase().replaceAll("[^a-z0-9]", "");
    }

    private boolean isCharacterSubstitutionAttempt(String str1, String str2) {
        String norm1 = str1.replace('0', 'o').replace('1', 'l').replace('5', 's');
        String norm2 = str2.replace('0', 'o').replace('1', 'l').replace('5', 's');
        return norm1.equals(norm2);
    }
}