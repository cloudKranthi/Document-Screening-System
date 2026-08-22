package com.sih.document_screening.model;

import lombok.Getter;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

@Getter
public enum VisaType {
    TOURIST("Tourist Visa", 90),
    BUSINESS("Business Visa", 180),
    STUDENT("Student Visa", 365),
    WORK_PERMIT("Work Permit Visa", 730),
    TRANSIT("Transit Visa", 5),
    DIPLOMATIC("Diplomatic Visa", 365);

    private final String description;
    private final int defaultMaxStayDays;

    private static final Map<String, VisaType> LOOKUP_MAP;

    static {
        Map<String, VisaType> map = new HashMap<>();
        for (VisaType type : VisaType.values()) {
            map.put(type.name(), type);
            // Allow case-insensitive lookups from OCR strings (e.g., "tourist", "BUSINESS_VISA")
            map.put(type.name().replace("_", ""), type);
            map.put(type.getDescription().toUpperCase(), type);
        }
        LOOKUP_MAP = Collections.unmodifiableMap(map);
    }

    VisaType(String description, int defaultMaxStayDays) {
        this.description = description;
        this.defaultMaxStayDays = defaultMaxStayDays;
    }

    public static VisaType fromString(String text) {
        if (text == null || text.trim().isEmpty()) {
            return null;
        }
        String normalized = text.trim().toUpperCase().replace(" ", "_");
        return LOOKUP_MAP.getOrDefault(normalized, null);
    }
}