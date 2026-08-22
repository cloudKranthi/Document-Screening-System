package com.sih.document_screening.model;

import java.time.LocalDate;
import java.util.UUID;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

@Entity
@Table(name="document_records")
public class DocumentRecord {
     @Id
    @GeneratedValue(strategy = GenerationType.UUID)
    private UUID id;

    @OneToOne
    @JoinColumn(name = "screening_id", nullable = false)
    private Screening screening;

    private String passportNumber;
    private String passportName;
    private LocalDate passportDob;
    private LocalDate passportExpiry;
    private String passportNationality;
    private Boolean mrzChecksumValid;

    private String visaNumber;
    private LocalDate visaValidUntil;
    private String visaType;


    private String nationalIdNumber;
    private String nationalIdName;
    private Boolean isNameConsistent;
    private Boolean isDobConsistent;
    private Boolean isNationalityConsistent;
    private Boolean isDocumentExpired;
}
